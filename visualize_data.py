#!/usr/bin/env python3
"""
Network Data Completeness Visualization
========================================

Creates before/after visualizations showing data completeness improvements
after extraction from obs column, broken down by product groups.

Usage:
    python visualize_data.py processed_data.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys
from pathlib import Path


class NetworkDataVisualizer:
    """Create visualizations for network data completeness analysis"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Product group configurations (same as extractor)
        self.product_groups = {
            "bandalarga_broadband_fiber_plans": [
                "serial_code", "model_onu", "wifi_ssid", "wifi_passcode", "vlan", "login_pppoe"
            ],
            "linkdeinternet_dedicated_internet_connectivity": [
                "client_type", "technology_id", "cpe", "ip_management", "vlan", "ip_block", "pop_description", "interface_1"
            ],
            "linkdeinternet_gpon_semi_dedicated_connections": [
                "client_type", "technology_id", "provider_id", "cpe", "vlan", "serial_code", "pop_description"
            ],
            "linkdeinternet_direct_l2l_links": [
                "client_type", "technology_id", "provider_id", "pop_description", "cpe", "interface_1", "vlan"
            ],
            "linkdeinternet_mpls_data_transport_services": [
                "client_type", "technology_id", "provider_id", "pop_description", "interface_1", "vlan", "cpe", "ip_management"
            ],
            "linkdeinternet_lan_to_lan_infrastructure": [
                "client_type", "technology_id", "provider_id", "pop_description", "interface_1", "cpe", "ip_management", "vlan"
            ],
            "linkdeinternet_ip_transit_services": [
                "client_type", "technology_id", "provider_id", "pop_description", "interface_1", "gateway", "asn", "vlan", "prefixes"
            ],
            "linkdeinternet_traffic_exchange_ptt": [
                "client_type", "technology_id", "provider_id", "pop_description", "interface_1", "vlan"
            ],
            "linkdeinternet_enterprise_gpon_lan": [
                "client_type", "technology_id", "provider_id", "pop_description", "serial_code", "cpe", "ip_management", "vlan"
            ]
        }
    
    def calculate_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate before/after completeness for each product group"""
        
        results = []
        
        for product_group, mandatory_fields in self.product_groups.items():
            # Filter data for this product group
            group_df = df[df['product_group'] == product_group].copy()
            
            if len(group_df) == 0:
                continue
            
            # Calculate BEFORE completeness (original columns)
            before_filled = 0
            before_total = len(group_df) * len(mandatory_fields)
            
            for field in mandatory_fields:
                if field in group_df.columns:
                    filled_count = group_df[field].notna().sum()
                    # Also check for non-empty strings
                    filled_count = group_df[field].apply(
                        lambda x: pd.notna(x) and str(x).strip() != ''
                    ).sum()
                    before_filled += filled_count
            
            before_completeness = (before_filled / before_total) * 100 if before_total > 0 else 0
            
            # Calculate AFTER completeness (extracted columns)
            after_filled = 0
            after_total = len(group_df) * len(mandatory_fields)
            
            for field in mandatory_fields:
                # Check both original and extracted columns
                original_filled = 0
                extracted_filled = 0
                
                if field in group_df.columns:
                    original_filled = group_df[field].apply(
                        lambda x: pd.notna(x) and str(x).strip() != ''
                    ).sum()
                
                extracted_field = f"extracted_{field}"
                if extracted_field in group_df.columns:
                    extracted_filled = group_df[extracted_field].apply(
                        lambda x: pd.notna(x) and str(x).strip() != ''
                    ).sum()
                
                # Count field as filled if either original OR extracted has data
                combined_filled = group_df.apply(lambda row: 
                    (pd.notna(row.get(field)) and str(row.get(field, '')).strip() != '') or
                    (pd.notna(row.get(extracted_field)) and str(row.get(extracted_field, '')).strip() != ''),
                    axis=1
                ).sum()
                
                after_filled += combined_filled
            
            after_completeness = (after_filled / after_total) * 100 if after_total > 0 else 0
            
            # Calculate improvement
            improvement = after_completeness - before_completeness
            
            results.append({
                'Product_Group': product_group.replace('_', '\n'),  # Break long names
                'Product_Group_Short': product_group.split('_')[0],  # For shorter labels
                'Records': len(group_df),
                'Mandatory_Fields': len(mandatory_fields),
                'Before_Completeness': before_completeness,
                'After_Completeness': after_completeness,
                'Improvement': improvement,
                'Before_Filled': before_filled,
                'After_Filled': after_filled,
                'Total_Possible': after_total
            })
        
        return pd.DataFrame(results)
    
    def create_completeness_comparison_bar(self, completeness_df: pd.DataFrame):
        """Create before/after completeness comparison bar chart"""
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(completeness_df))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, completeness_df['Before_Completeness'], 
                      width, label='Before Extraction', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, completeness_df['After_Completeness'], 
                      width, label='After Extraction', color='#2ecc71', alpha=0.8)
        
        # Add value labels on bars
        for i, (before, after) in enumerate(zip(completeness_df['Before_Completeness'], 
                                               completeness_df['After_Completeness'])):
            ax.text(i - width/2, before + 1, f'{before:.1f}%', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax.text(i + width/2, after + 1, f'{after:.1f}%', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Customize chart
        ax.set_xlabel('Product Groups', fontweight='bold', fontsize=12)
        ax.set_ylabel('Completeness Percentage', fontweight='bold', fontsize=12)
        ax.set_title('Data Completeness: Before vs After Extraction\nby Product Group', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(completeness_df['Product_Group_Short'], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'completeness_comparison_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_improvement_histogram(self, completeness_df: pd.DataFrame):
        """Create histogram of improvements by product group"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of improvements
        ax1.hist(completeness_df['Improvement'], bins=10, color='#3498db', alpha=0.8, edgecolor='black')
        ax1.axvline(completeness_df['Improvement'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {completeness_df["Improvement"].mean():.1f}%')
        ax1.set_xlabel('Improvement in Completeness (%)', fontweight='bold')
        ax1.set_ylabel('Number of Product Groups', fontweight='bold')
        ax1.set_title('Distribution of Data Completeness Improvements', fontweight='bold', pad=15)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Bar chart of individual improvements
        bars = ax2.bar(range(len(completeness_df)), completeness_df['Improvement'], 
                      color='#9b59b6', alpha=0.8)
        
        # Color bars by improvement level
        for i, bar in enumerate(bars):
            improvement = completeness_df.iloc[i]['Improvement']
            if improvement > 20:
                bar.set_color('#2ecc71')  # Green for high improvement
            elif improvement > 10:
                bar.set_color('#f39c12')  # Orange for medium improvement
            else:
                bar.set_color('#e74c3c')  # Red for low improvement
        
        # Add value labels
        for i, improvement in enumerate(completeness_df['Improvement']):
            ax2.text(i, improvement + 0.5, f'{improvement:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Product Groups', fontweight='bold')
        ax2.set_ylabel('Improvement (%)', fontweight='bold')
        ax2.set_title('Completeness Improvement by Product Group', fontweight='bold', pad=15)
        ax2.set_xticks(range(len(completeness_df)))
        ax2.set_xticklabels(completeness_df['Product_Group_Short'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_records_distribution_pie(self, completeness_df: pd.DataFrame):
        """Create pie charts showing record distribution and completeness"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart 1: Distribution of records by product group
        colors1 = sns.color_palette("husl", len(completeness_df))
        wedges1, texts1, autotexts1 = ax1.pie(completeness_df['Records'], 
                                             labels=completeness_df['Product_Group_Short'],
                                             autopct='%1.1f%%',
                                             colors=colors1,
                                             startangle=90)
        
        ax1.set_title('Distribution of Records by Product Group', fontweight='bold', fontsize=12, pad=20)
        
        # Make percentage text bold
        for autotext in autotexts1:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Pie chart 2: Average completeness after extraction
        colors2 = ['#2ecc71' if x > 80 else '#f39c12' if x > 60 else '#e74c3c' 
                  for x in completeness_df['After_Completeness']]
        
        wedges2, texts2, autotexts2 = ax2.pie(completeness_df['After_Completeness'], 
                                             labels=completeness_df['Product_Group_Short'],
                                             autopct='%1.1f%%',
                                             colors=colors2,
                                             startangle=90)
        
        ax2.set_title('Data Completeness After Extraction\nby Product Group', 
                     fontweight='bold', fontsize=12, pad=20)
        
        # Make percentage text bold
        for autotext in autotexts2:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_pie_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_field_analysis(self, df: pd.DataFrame, completeness_df: pd.DataFrame):
        """Create detailed analysis of field-level completeness"""
        
        # Select top 3 product groups by record count for detailed analysis
        top_groups = completeness_df.nlargest(3, 'Records')['Product_Group_Short'].str.replace('\n', '_')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (ax, group_short) in enumerate(zip(axes, top_groups)):
            # Find the full group name
            full_group_name = None
            for full_name in self.product_groups.keys():
                if full_name.startswith(group_short):
                    full_group_name = full_name
                    break
            
            if not full_group_name:
                continue
                
            group_df = df[df['product_group'] == full_group_name]
            mandatory_fields = self.product_groups[full_group_name]
            
            field_completeness = []
            field_names = []
            
            for field in mandatory_fields:
                # Calculate combined completeness (original + extracted)
                original_filled = 0
                extracted_filled = 0
                
                if field in group_df.columns:
                    original_filled = group_df[field].apply(
                        lambda x: pd.notna(x) and str(x).strip() != ''
                    ).sum()
                
                extracted_field = f"extracted_{field}"
                if extracted_field in group_df.columns:
                    extracted_filled = group_df[extracted_field].apply(
                        lambda x: pd.notna(x) and str(x).strip() != ''
                    ).sum()
                
                # Combined (either original OR extracted has data)
                combined_filled = group_df.apply(lambda row: 
                    (pd.notna(row.get(field)) and str(row.get(field, '')).strip() != '') or
                    (pd.notna(row.get(extracted_field)) and str(row.get(extracted_field, '')).strip() != ''),
                    axis=1
                ).sum()
                
                completeness_pct = (combined_filled / len(group_df)) * 100
                field_completeness.append(completeness_pct)
                field_names.append(field.replace('_', '\n'))
            
            # Create horizontal bar chart
            y_pos = np.arange(len(field_names))
            bars = ax.barh(y_pos, field_completeness, color='#3498db', alpha=0.8)
            
            # Color bars by completeness level
            for i, bar in enumerate(bars):
                if field_completeness[i] > 80:
                    bar.set_color('#2ecc71')
                elif field_completeness[i] > 60:
                    bar.set_color('#f39c12')
                else:
                    bar.set_color('#e74c3c')
            
            # Add percentage labels
            for i, pct in enumerate(field_completeness):
                ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(field_names, fontsize=9)
            ax.set_xlabel('Completeness (%)', fontweight='bold')
            ax.set_title(f'{group_short.title()}\nField Completeness', fontweight='bold', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'field_level_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_dashboard(self, completeness_df: pd.DataFrame):
        """Create a summary dashboard with key metrics"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall improvement bar
        ax1 = fig.add_subplot(gs[0, :])
        total_before = completeness_df['Before_Completeness'].mean()
        total_after = completeness_df['After_Completeness'].mean()
        
        bars = ax1.bar(['Before Extraction', 'After Extraction'], [total_before, total_after], 
                      color=['#e74c3c', '#2ecc71'], alpha=0.8, width=0.5)
        
        for i, (bar, value) in enumerate(zip(bars, [total_before, total_after])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax1.set_ylabel('Average Completeness (%)', fontweight='bold')
        ax1.set_title('Overall Data Completeness Improvement', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Records by product group
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(range(len(completeness_df)), completeness_df['Records'], 
               color='#9b59b6', alpha=0.8)
        ax2.set_title('Records by\nProduct Group', fontweight='bold')
        ax2.set_xlabel('Product Groups')
        ax2.set_ylabel('Record Count')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Improvement distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(completeness_df['Improvement'], bins=8, color='#3498db', alpha=0.8)
        ax3.set_title('Improvement\nDistribution', fontweight='bold')
        ax3.set_xlabel('Improvement (%)')
        ax3.set_ylabel('Frequency')
        
        # Top performers
        ax4 = fig.add_subplot(gs[1, 2])
        top_3 = completeness_df.nlargest(3, 'After_Completeness')
        ax4.barh(range(len(top_3)), top_3['After_Completeness'], 
                color='#2ecc71', alpha=0.8)
        ax4.set_yticks(range(len(top_3)))
        ax4.set_yticklabels(top_3['Product_Group_Short'], fontsize=8)
        ax4.set_title('Top 3 Product Groups\n(After Extraction)', fontweight='bold')
        ax4.set_xlabel('Completeness (%)')
        
        # Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary text
        total_records = completeness_df['Records'].sum()
        avg_improvement = completeness_df['Improvement'].mean()
        best_group = completeness_df.loc[completeness_df['After_Completeness'].idxmax(), 'Product_Group_Short']
        worst_group = completeness_df.loc[completeness_df['After_Completeness'].idxmin(), 'Product_Group_Short']
        
        summary_text = f"""
        📊 EXTRACTION SUMMARY STATISTICS
        
        Total Records Processed: {total_records:,}
        Average Improvement: {avg_improvement:.1f} percentage points
        Best Performing Group: {best_group} ({completeness_df['After_Completeness'].max():.1f}% complete)
        Needs Most Attention: {worst_group} ({completeness_df['After_Completeness'].min():.1f}% complete)
        
        Product Groups with >80% Completeness: {(completeness_df['After_Completeness'] > 80).sum()}/{len(completeness_df)}
        """
        
        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Network Data Extraction Analysis Dashboard', 
                    fontweight='bold', fontsize=16, y=0.95)
        
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self, csv_file: str):
        """Generate all visualizations from the processed CSV"""
        
        print(f"📊 Loading data from: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return
        
        print(f"📈 Analyzing {len(df)} records across {df['product_group'].nunique()} product groups")
        
        # Calculate completeness metrics
        completeness_df = self.calculate_completeness(df)
        
        if len(completeness_df) == 0:
            print("❌ No data found for analysis")
            return
        
        print("🎨 Generating visualizations...")
        
        # Generate all charts
        self.create_completeness_comparison_bar(completeness_df)
        self.create_improvement_histogram(completeness_df)
        self.create_records_distribution_pie(completeness_df)
        self.create_detailed_field_analysis(df, completeness_df)
        self.create_summary_dashboard(completeness_df)
        
        print(f"✅ All visualizations saved to: {self.output_dir}")
        
        # Print summary
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        print(f"Average completeness before: {completeness_df['Before_Completeness'].mean():.1f}%")
        print(f"Average completeness after: {completeness_df['After_Completeness'].mean():.1f}%")
        print(f"Average improvement: {completeness_df['Improvement'].mean():.1f} percentage points")
        print(f"Best performing group: {completeness_df.loc[completeness_df['After_Completeness'].idxmax(), 'Product_Group_Short']}")
        print(f"Total records analyzed: {completeness_df['Records'].sum():,}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize network data extraction results')
    parser.add_argument('csv_file', help='Processed CSV file from network extractor')
    parser.add_argument('-o', '--output', help='Output directory for visualizations', default='visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer and generate charts
    visualizer = NetworkDataVisualizer(args.output)
    visualizer.generate_all_visualizations(args.csv_file)


if __name__ == "__main__":
    main()