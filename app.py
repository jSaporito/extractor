import os
import io
import base64
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import uuid
load_dotenv()

from network_extractor import NetworkConfigExtractor

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def safe_jsonify(data):
    """Custom jsonify that handles numpy/pandas types"""
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    converted_data = convert_types(data)
    return jsonify(converted_data)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
CHUNK_SIZE = 1000  # Process in chunks of 1000 rows for large files

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

# Global storage for processing status
processing_status = {}

class ProcessingManager:
    """Manage background processing tasks"""
    
    def __init__(self):
        self.extractor = NetworkConfigExtractor()
    
    def process_file_chunked(self, file_path, task_id, chunk_size=CHUNK_SIZE):
        """Process large files in chunks with progress tracking"""
        try:
            processing_status[task_id] = {
                'status': 'reading',
                'progress': 0,
                'message': 'Reading CSV file...',
                'start_time': datetime.now()
            }
            
            # Read the file
            df = pd.read_csv(file_path)
            total_rows = len(df)
            
            processing_status[task_id].update({
                'status': 'processing',
                'message': f'Processing {total_rows} rows...',
                'total_rows': total_rows
            })
            
            # Check if chunking is needed
            if total_rows <= chunk_size:
                return self.process_single_chunk(df, task_id, file_path)
            else:
                return self.process_multiple_chunks(df, task_id, file_path, chunk_size)
                
        except Exception as e:
            processing_status[task_id].update({
                'status': 'error',
                'message': f'Error: {str(e)}'
            })
            return None
    
    def process_single_chunk(self, df, task_id, original_file_path):
        """Process file as single chunk"""
        try:
            # Extract data
            all_extracted_columns = set()
            for idx, row in df.iterrows():
                product_group = str(row.get('product_group', ''))
                obs_text = row.get('obs', '')
                
                if product_group in self.extractor.product_groups and pd.notna(obs_text):
                    extracted = self.extractor.extract_for_product_group(obs_text, product_group)
                    
                    for key, value in extracted.items():
                        df.at[idx, key] = value
                        all_extracted_columns.add(key)
                
                # Update progress
                progress = ((idx + 1) / len(df)) * 80  # 80% for extraction
                processing_status[task_id]['progress'] = progress
                processing_status[task_id]['message'] = f'Processing row {idx + 1}/{len(df)}'
            
            return self.finalize_processing(df, task_id, original_file_path)
            
        except Exception as e:
            processing_status[task_id].update({
                'status': 'error',
                'message': f'Processing error: {str(e)}'
            })
            return None
    
    def process_multiple_chunks(self, df, task_id, original_file_path, chunk_size):
        """Process file in multiple chunks"""
        try:
            total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(df))
                
                chunk_df = df.iloc[start_idx:end_idx].copy()
                
                # Process this chunk
                for idx, row in chunk_df.iterrows():
                    product_group = str(row.get('product_group', ''))
                    obs_text = row.get('obs', '')
                    
                    if product_group in self.extractor.product_groups and pd.notna(obs_text):
                        extracted = self.extractor.extract_for_product_group(obs_text, product_group)
                        
                        for key, value in extracted.items():
                            df.at[idx, key] = value
                
                # Update progress
                progress = ((chunk_idx + 1) / total_chunks) * 80
                processing_status[task_id]['progress'] = progress
                processing_status[task_id]['message'] = f'Processing chunk {chunk_idx + 1}/{total_chunks}'
            
            return self.finalize_processing(df, task_id, original_file_path)
            
        except Exception as e:
            processing_status[task_id].update({
                'status': 'error',
                'message': f'Chunk processing error: {str(e)}'
            })
            return None
    
    def finalize_processing(self, df, task_id, original_file_path):
        """Add validation and save results"""
        try:
            processing_status[task_id].update({
                'progress': 85,
                'message': 'Adding validation...'
            })
            
            # Add validation
            validation_results = []
            for idx, row in df.iterrows():
                product_group = str(row.get('product_group', ''))
                validation = self.extractor.validate_extraction(row, product_group)
                validation_results.append(validation)
            
            df['validation_is_complete'] = [v['is_valid'] for v in validation_results]
            df['validation_completion_pct'] = [v['completion'] for v in validation_results]
            df['validation_missing_fields'] = ['; '.join(v['missing']) for v in validation_results]
            df['validation_filled_count'] = [v['filled_count'] for v in validation_results]
            
            processing_status[task_id].update({
                'progress': 95,
                'message': 'Saving results...'
            })
            
            # Save outputs
            base_filename = Path(original_file_path).stem
            csv_output = Path(OUTPUT_FOLDER) / f"{base_filename}_processed_{task_id}.csv"
            excel_output = Path(OUTPUT_FOLDER) / f"{base_filename}_processed_{task_id}.xlsx"
            
            # Save CSV
            df.to_csv(csv_output, index=False)
            
            # Save Excel with summary
            with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Network_Config', index=False)
                
                # Create summary
                summary_data = []
                for pg in df['product_group'].value_counts().index:
                    if pd.notna(pg):
                        pg_df = df[df['product_group'] == pg]
                        complete_count = pg_df['validation_is_complete'].sum()
                        avg_completion = pg_df['validation_completion_pct'].mean()
                        
                        summary_data.append({
                            'Product_Group': pg,
                            'Total_Records': len(pg_df),
                            'Complete_Records': complete_count,
                            'Completion_Rate': f"{complete_count/len(pg_df)*100:.1f}%",
                            'Avg_Completion': f"{avg_completion:.1f}%"
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Generate visualization
            chart_data = self.create_visualization(df, task_id)
            
            processing_status[task_id].update({
                'status': 'completed',
                'progress': 100,
                'message': 'Processing completed successfully!',
                'csv_file': str(csv_output),
                'excel_file': str(excel_output),
                'chart_data': chart_data,
                'end_time': datetime.now(),
                'summary': {
                    'total_rows': int(len(df)),
                    'complete_rows': int(df['validation_is_complete'].sum()),
                    'avg_completion': float(df['validation_completion_pct'].mean()),
                    'product_groups': int(df['product_group'].nunique())
                }
            })
            
            return df
            
        except Exception as e:
            processing_status[task_id].update({
                'status': 'error',
                'message': f'Finalization error: {str(e)}'
            })
            return None
    
    def create_visualization(self, df, task_id):
        """Create visualization charts"""
        try:
            # Calculate completeness by product group
            product_groups = df['product_group'].value_counts().index
            completeness_data = []
            
            for pg in product_groups:
                if pd.notna(pg):
                    pg_df = df[df['product_group'] == pg]
                    avg_completion = float(pg_df['validation_completion_pct'].mean())
                    record_count = int(len(pg_df))
                    
                    completeness_data.append({
                        'Product_Group': pg.replace('_', ' ').title(),
                        'Avg_Completion': avg_completion,
                        'Record_Count': record_count
                    })
            
            # Create bar chart
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Completeness chart
            groups = [item['Product_Group'][:20] + '...' if len(item['Product_Group']) > 20 
                     else item['Product_Group'] for item in completeness_data]
            completeness = [item['Avg_Completion'] for item in completeness_data]
            
            bars1 = ax1.bar(range(len(groups)), completeness, color='#2ecc71', alpha=0.8)
            ax1.set_xlabel('Product Groups')
            ax1.set_ylabel('Average Completeness (%)')
            ax1.set_title('Data Completeness by Product Group')
            ax1.set_xticks(range(len(groups)))
            ax1.set_xticklabels(groups, rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Record count chart
            record_counts = [item['Record_Count'] for item in completeness_data]
            bars2 = ax2.bar(range(len(groups)), record_counts, color='#3498db', alpha=0.8)
            ax2.set_xlabel('Product Groups')
            ax2.set_ylabel('Number of Records')
            ax2.set_title('Record Distribution by Product Group')
            ax2.set_xticks(range(len(groups)))
            ax2.set_xticklabels(groups, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = Path(OUTPUT_FOLDER) / f"chart_{task_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Convert to base64 for web display
            with open(chart_path, "rb") as img_file:
                chart_b64 = base64.b64encode(img_file.read()).decode()
            
            return {
                'chart_b64': chart_b64,
                'chart_path': str(chart_path),
                'completeness_data': completeness_data
            }
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None

# Global processing manager
processor = ProcessingManager()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = Path(UPLOAD_FOLDER) / unique_filename
        file.save(file_path)
        
        # Generate task ID
        task_id = str(uuid.uuid4())[:8]
        session['current_task'] = task_id
        
        # Start processing in background thread
        thread = threading.Thread(
            target=processor.process_file_chunked,
            args=(file_path, task_id)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'File uploaded successfully. Processing started.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status"""
    if task_id in processing_status:
        status = processing_status[task_id].copy()
        
        # Calculate elapsed time
        if 'start_time' in status:
            elapsed = datetime.now() - status['start_time']
            status['elapsed_time'] = str(elapsed).split('.')[0]  # Remove microseconds
        
        return safe_jsonify(status)
    else:
        return safe_jsonify({'error': 'Task not found'}), 404

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download processed files"""
    try:
        file_path = Path(OUTPUT_FOLDER) / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/results/<task_id>')
def view_results(task_id):
    """View processing results"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    if status.get('status') != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    return render_template('results.html', 
                         task_id=task_id, 
                         status=status)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)