import pandas as pd
import re
import sys
import argparse
from typing import Dict, List, Optional, Any


class NetworkConfigExtractor:
    
    def __init__(self):
        self.ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'
        self.vlan_pattern = r'(?:VLAN|vlan)[:\s]*(\d+)'
        self.mac_pattern = r'(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}'
        
        # Product group mandatory fields mapping
        self.product_groups = {
            "bandalarga_broadband_fiber_plans": {
                "fields": ["serial_code", "model_onu", "wifi_ssid", "wifi_passcode", "vlan", "login_pppoe"],
                "extractors": {
                    "serial_code": ["serial_code", "serial", "mac"],
                    "model_onu": ["model_onu", "terminal_type", "onu_model"],
                    "wifi_ssid": ["wifi_ssid", "ssid"],
                    "wifi_passcode": ["wifi_passcode", "wifi_password"],
                    "vlan": ["vlan", "vlan_config"],
                    "login_pppoe": ["login_pppoe", "pppoe_user", "username"]
                }
            },
            "linkdeinternet_dedicated_internet_connectivity": {
                "fields": ["client_type", "technology_id", "cpe", "ip_management", "vlan", "ip_block", "pop_description", "interface_1"],
                "extractors": {
                    "cpe": ["cpe", "router", "equipment"],
                    "ip_management": ["ip_management", "ip_privado", "private_ip"],
                    "vlan": ["vlan", "vlan_config"],
                    "ip_block": ["ip_block", "ip_publico", "public_ip"],
                    "pop_description": ["pop_description", "pop", "location"],
                    "interface_1": ["interface_1", "interface", "port"]
                }
            },
            "linkdeinternet_gpon_semi_dedicated_connections": {
                "fields": ["client_type", "technology_id", "provider_id", "cpe", "vlan", "serial_code", "pop_description"],
                "extractors": {
                    "cpe": ["cpe", "onu", "equipment"],
                    "vlan": ["vlan", "vlan_config"],
                    "serial_code": ["serial_code", "serial", "onu_serial"],
                    "pop_description": ["pop_description", "pop", "olt_info"]
                }
            },
            "linkdeinternet_direct_l2l_links": {
                "fields": ["client_type", "technology_id", "provider_id", "pop_description", "cpe", "interface_1", "vlan"],
                "extractors": {
                    "pop_description": ["pop_description", "pop", "location"],
                    "cpe": ["cpe", "equipment"],
                    "interface_1": ["interface_1", "interface", "port"],
                    "vlan": ["vlan", "vlan_config"]
                }
            },
            "linkdeinternet_mpls_data_transport_services": {
                "fields": ["client_type", "technology_id", "provider_id", "pop_description", "interface_1", "vlan", "cpe", "ip_management"],
                "extractors": {
                    "pop_description": ["pop_description", "pop", "location"],
                    "interface_1": ["interface_1", "interface", "port"],
                    "vlan": ["vlan", "vlan_config"],
                    "cpe": ["cpe", "equipment"],
                    "ip_management": ["ip_management", "ip_privado", "management_ip"]
                }
            },
            "linkdeinternet_lan_to_lan_infrastructure": {
                "fields": ["client_type", "technology_id", "provider_id", "pop_description", "interface_1", "cpe", "ip_management", "vlan"],
                "extractors": {
                    "pop_description": ["pop_description", "pop", "location"],
                    "interface_1": ["interface_1", "interface", "port"],
                    "cpe": ["cpe", "equipment"],
                    "ip_management": ["ip_management", "ip_privado", "management_ip"],
                    "vlan": ["vlan", "vlan_config"]
                }
            },
            "linkdeinternet_ip_transit_services": {
                "fields": ["client_type", "technology_id", "provider_id", "pop_description", "interface_1", "gateway", "asn", "vlan", "prefixes"],
                "extractors": {
                    "pop_description": ["pop_description", "pop", "location"],
                    "interface_1": ["interface_1", "interface", "port"],
                    "gateway": ["gateway", "default_gateway"],
                    "asn": ["asn", "as_number"],
                    "vlan": ["vlan", "vlan_config"],
                    "prefixes": ["prefixes", "ip_block", "routes"]
                }
            },
            "linkdeinternet_traffic_exchange_ptt": {
                "fields": ["client_type", "technology_id", "provider_id", "pop_description", "interface_1", "vlan"],
                "extractors": {
                    "pop_description": ["pop_description", "pop", "location"],
                    "interface_1": ["interface_1", "interface", "port"],
                    "vlan": ["vlan", "vlan_config"]
                }
            },
            "linkdeinternet_enterprise_gpon_lan": {
                "fields": ["client_type", "technology_id", "provider_id", "pop_description", "serial_code", "cpe", "ip_management", "vlan"],
                "extractors": {
                    "pop_description": ["pop_description", "pop", "olt_info"],
                    "serial_code": ["serial_code", "serial", "onu_serial"],
                    "cpe": ["cpe", "onu", "equipment"],
                    "ip_management": ["ip_management", "ip_privado", "management_ip"],
                    "vlan": ["vlan", "vlan_config"]
                }
            }
        }
    
    def clean_obs_text(self, text: str) -> str:
        """Remove noise from obs text while preserving important data"""
        if not text or pd.isna(text):
            return ""
        
        clean_text = str(text).replace('\r\n', '\n').replace('\r', '\n')
        
        noise_patterns = [
            r'-{10,}',  # Long dashes
            r'={10,}',  # Long equals
            r'_{10,}',  # Long underscores
            r'^\s*#.*$',  # Comment lines
            r'^\s*\[.*?\]\s*>\s*$',  # Command prompts
            r'/tool fetch.*?(?=\n|$)',  # Router config commands
            r'/system.*?(?=\n|$)',
            r'set \[.*?\].*?(?=\n|$)',
            r'add .*?policy=.*?(?=\n|$)',
            r'cfgcli -s.*?(?=\n|$)',
            r'\\\\.*?\\\\',  # Escaped strings
            r'\\".*?\\"',    # Quoted strings with escapes
        ]
        
        for pattern in noise_patterns:
            clean_text = re.sub(pattern, ' ', clean_text, flags=re.MULTILINE | re.IGNORECASE)
        
        clean_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_text)  # Multiple blank lines
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Multiple spaces
        clean_text = clean_text.strip()
        
        return clean_text
    
    def extract_value(self, text: str, field_type: str) -> Optional[str]:
        """Extract specific field value from cleaned text"""
        if not text:
            return None
        
        patterns = {
            "designador": [r'(?:DESIGNADOR|Designador)[:\s]*(.*?)(?=\n|$)'],
            "ip_privado": [r'(?:IPPRIVADO|IP PRIVADO)[:\s]*((?:\d+\..*?\n?)*)', 
                          r'(?:Private IP|PRIVATE)[:\s]*((?:\d+\..*?\n?)*)'],
            "ip_publico": [r'(?:IPPUBLICO|IP PUBLICO)[:\s]*((?:\d+\..*?\n?)*)',
                          r'(?:Public IP|PUBLIC)[:\s]*((?:\d+\..*?\n?)*)'],
            "ip_management": [r'(?:IP MANAGEMENT|MANAGEMENT)[:\s]*((?:\d+\..*?\n?)*)',
                             r'(?:MGMT IP)[:\s]*((?:\d+\..*?\n?)*)'],
            "ip_block": [r'(?:IP BLOCK|BLOCK)[:\s]*((?:\d+\..*?\n?)*)',
                        r'(?:Network|NETWORK)[:\s]*((?:\d+\..*?\n?)*)'],
            "fixed_ip": [r'(?:IP FIXO|Fixed IP)[:\s]*([0-9.]+(?:/\d+)?)'],
            
            "vlan": [r'(?:VLAN)[:\s]*(\d+)', r'(?:vlan)[:\s]*(\d+)'],
            "vlan_config": [r'(?:VLAN)[:\s]*(\d+)', r'(?:vlan)[:\s]*(\d+)'],
            
            "pppoe_user": [r'(?:USER PPPOE|PPPoE USER)[:\s]*([\w_]+)',
                          r'(?:Username)[:\s]*([\w_]+)'],
            "username": [r'(?:USER PPPOE|PPPoE USER|Username)[:\s]*([\w_]+)'],
            "login_pppoe": [r'(?:USER PPPOE|PPPoE USER|Username)[:\s]*([\w_]+)'],
            "pppoe_password": [r'(?:SENHA|Password)[:\s]*([^\s\n]+)'],
            
            "serial_code": [r'(?:Serial|SN|SERNUM)[:\s=]*([0-9A-Fa-f]+)',
                           r'(?:serial_code)[:\s]*([^\s\n]+)'],
            "serial": [r'(?:Serial|SN|SERNUM)[:\s=]*([0-9A-Fa-f]+)'],
            "onu_serial": [r'(?:Serial|SN|SERNUM)[:\s=]*([0-9A-Fa-f]+)'],
            "mac": [r'((?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2})'],
            
            "model_onu": [r'(?:Terminal Type|Model)[:\s]*([^\n\t]+)',
                         r'(?:ONU Model)[:\s]*([^\n\t]+)'],
            "terminal_type": [r'(?:Terminal Type)[:\s]*([^\n\t]+)'],
            "onu_model": [r'(?:ONU Model|Model)[:\s]*([^\n\t]+)'],
            
            "cpe": [r'(?:CPE)[:\s]*([^\n\t]+)', 
                   r'(?:Router)[:\s]*([^\n\t]+)',
                   r'(?:Equipment)[:\s]*([^\n\t]+)'],
            "equipment": [r'(?:Equipment|CPE|Router)[:\s]*([^\n\t]+)'],
            "router": [r'(?:Router)[:\s]*([^\n\t]+)'],
            "onu": [r'(?:ONU)[:\s]*([^\n\t]+)'],
            
            
            "interface_1": [r'(?:Interface|Port)[:\s]*([^\n\t]+)',
                           r'(?:interface_1)[:\s]*([^\n\t]+)'],
            "interface": [r'(?:Interface|Port)[:\s]*([^\n\t]+)'],
            "port": [r'(?:Port)[:\s]*([^\n\t]+)'],
            
            "gateway": [r'(?:Gateway|Default Gateway)[:\s]*([0-9.]+)'],
            "default_gateway": [r'(?:Gateway|Default Gateway)[:\s]*([0-9.]+)'],
            
            "asn": [r'(?:ASN|AS Number)[:\s]*(\d+)'],
            "as_number": [r'(?:ASN|AS Number)[:\s]*(\d+)'],
            
            "prefixes": [r'(?:Prefixes|Routes)[:\s]*([^\n]+)'],
            "routes": [r'(?:Routes|Prefixes)[:\s]*([^\n]+)'],
            
            
            "pop_description": [r'(?:POP|Location)[:\s]*([^\n]+)',
                               r'(?:pop_description)[:\s]*([^\n]+)'],
            "pop": [r'(?:POP|Location)[:\s]*([^\n]+)'],
            "location": [r'(?:Location|POP)[:\s]*([^\n]+)'],
            "olt_info": [r'(?:OLT)[:\s-]*([^\n]+)', r'(?:NE\s+)([^\n]+)'],
            
            
            "wifi_ssid": [r'(?:SSID|WiFi SSID)[:\s]*([^\n]+)',
                         r'(?:wifi_ssid)[:\s]*([^\n]+)'],
            "ssid": [r'(?:SSID)[:\s]*([^\n]+)'],
            "wifi_passcode": [r'(?:WiFi Password|WiFi Pass|Password)[:\s]*([^\n]+)',
                             r'(?:wifi_passcode)[:\s]*([^\n]+)'],
            "wifi_password": [r'(?:WiFi Password|WiFi Pass)[:\s]*([^\n]+)'],
            
            "rate": [r'(?:RATE)[:\s]*([^\n]+)'],
            
            "chamado": [r'(?:CHAMADO)[:\s]*([^\n]+)'],
        }
        
        if field_type in patterns:
            for pattern in patterns[field_type]:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Try to get group 1, fall back to group 0 if no groups
                    try:
                        result = match.group(1).strip()
                    except IndexError:
                        result = match.group(0).strip()
                    
                    # Clean up the result
                    if result and result != '':
                        # For IP lists, join multiple IPs
                        if 'ip_' in field_type and '\n' in result:
                            ips = re.findall(self.ip_pattern, result)
                            return '; '.join(ips) if ips else result
                        return result
        
        return None
    
    def extract_for_product_group(self, obs_text: str, product_group: str) -> Dict[str, Any]:
        if not obs_text or pd.isna(obs_text) or product_group not in self.product_groups:
            return {}
        
        # Clean the text
        clean_text = self.clean_obs_text(obs_text)
        
        # Get mandatory fields for this product group
        group_config = self.product_groups[product_group]
        mandatory_fields = group_config["fields"]
        extractors = group_config.get("extractors", {})
        
        extracted = {}
        
        # Extract each mandatory field
        for field in mandatory_fields:
            value = None
            
            # Try extractors specific to this field
            if field in extractors:
                for extractor in extractors[field]:
                    value = self.extract_value(clean_text, extractor)
                    if value:
                        break
            
            # If no specific extractor worked, try the field name itself
            if not value:
                value = self.extract_value(clean_text, field)
            
            extracted[f"extracted_{field}"] = value
        
        # Add metadata
        extracted["cleaned_text_length"] = len(clean_text)
        extracted["original_text_length"] = len(obs_text)
        extracted["compression_ratio"] = round(len(clean_text) / len(obs_text), 2) if len(obs_text) > 0 else 0
        
        return extracted
    
    def validate_extraction(self, row: pd.Series, product_group: str) -> Dict[str, Any]:
        """Validate extracted data completeness"""
        if product_group not in self.product_groups:
            return {"is_valid": False, "completion": 0, "missing": []}
        
        mandatory_fields = self.product_groups[product_group]["fields"]
        filled_fields = []
        missing_fields = []
        
        for field in mandatory_fields:
            extracted_field = f"extracted_{field}"
            original_field = field
            
            # Check both extracted and original columns
            extracted_value = row.get(extracted_field)
            original_value = row.get(original_field)
            
            if (extracted_value and pd.notna(extracted_value) and str(extracted_value).strip()) or \
               (original_value and pd.notna(original_value) and str(original_value).strip()):
                filled_fields.append(field)
            else:
                missing_fields.append(field)
        
        completion = (len(filled_fields) / len(mandatory_fields)) * 100
        
        return {
            "is_valid": len(missing_fields) == 0,
            "completion": completion,
            "missing": missing_fields,
            "filled_count": len(filled_fields),
            "total_mandatory": len(mandatory_fields)
        }
    
    def process_csv(self, input_file: str, output_file: Optional[str] = None, excel: bool = False) -> Optional[pd.DataFrame]:
        """Process CSV file and extract network configuration data"""
        
        print(f"📖 Reading CSV file: {input_file}")
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None
        
        print(f"📊 Total rows: {len(df)}")
        
        # Check required columns
        if 'obs' not in df.columns:
            print("❌ Error: 'obs' column not found")
            return None
        
        if 'product_group' not in df.columns:
            print("❌ Error: 'product_group' column not found")
            return None
        
        # Stats
        obs_count = df['obs'].notna().sum()
        product_groups = df['product_group'].value_counts()
        
        print(f"📈 Rows with obs data: {obs_count} ({obs_count/len(df)*100:.1f}%)")
        print(f"🏷️  Product groups found: {len(product_groups)}")
        
        # Process each row based on its product group
        print("🔍 Extracting data by product group...")
        
        all_extracted_columns = set()
        for idx, row in df.iterrows():
            product_group = str(row.get('product_group', ''))
            obs_text = row.get('obs', '')
            
            if product_group in self.product_groups and pd.notna(obs_text):
                extracted = self.extract_for_product_group(obs_text, product_group)
                
                # Add extracted data to the row
                for key, value in extracted.items():
                    df.at[idx, key] = value
                    all_extracted_columns.add(key)
        
        # Add validation columns
        print("✅ Validating extracted data...")
        validation_results = []
        for idx, row in df.iterrows():
            product_group = str(row.get('product_group', ''))
            validation = self.validate_extraction(row, product_group)
            validation_results.append(validation)
        
        df['validation_is_complete'] = [v['is_valid'] for v in validation_results]
        df['validation_completion_pct'] = [v['completion'] for v in validation_results]
        df['validation_missing_fields'] = ['; '.join(v['missing']) for v in validation_results]
        df['validation_filled_count'] = [v['filled_count'] for v in validation_results]
        
        # Export results
        if output_file:
            if excel:
                print(f"💾 Saving to Excel: {output_file}")
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Main data
                    df.to_excel(writer, sheet_name='Network_Config', index=False)
                    
                    # Summary by product group
                    summary_data = []
                    for pg in product_groups.index:
                        if pd.notna(pg):
                            pg_df = df[df['product_group'] == pg]
                            complete_count = pg_df['validation_is_complete'].sum()
                            avg_completion = pg_df['validation_completion_pct'].mean()
                            
                            summary_data.append({
                                'Product_Group': pg,
                                'Total_Records': len(pg_df),
                                'Complete_Records': complete_count,
                                'Completion_Rate': f"{complete_count/len(pg_df)*100:.1f}%",
                                'Avg_Completion': f"{avg_completion:.1f}%",
                                'Mandatory_Fields': ', '.join(self.product_groups.get(str(pg), {}).get('fields', []))
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            else:
                print(f"💾 Saving to CSV: {output_file}")
                df.to_csv(output_file, index=False)
        
        return df
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print processing summary"""
        print("\n" + "="*60)
        print("🎯 EXTRACTION SUMMARY")
        print("="*60)
        
        total_rows = len(df)
        complete_rows = df.get('validation_is_complete', pd.Series(dtype=bool)).sum()
        avg_completion = df.get('validation_completion_pct', pd.Series(dtype=float)).mean()
        
        print(f"📊 Total rows processed: {total_rows}")
        print(f"✅ Complete records: {complete_rows} ({complete_rows/total_rows*100:.1f}%)")
        print(f"📈 Average completion: {avg_completion:.1f}%")
        
        # Product group breakdown
        if 'product_group' in df.columns:
            print(f"\n🏷️  Product Group Breakdown:")
            print("-" * 50)
            
            for product_group in df['product_group'].value_counts().index:
                if pd.notna(product_group):
                    pg_df = df[df['product_group'] == product_group]
                    pg_complete = pg_df.get('validation_is_complete', pd.Series(dtype=bool)).sum()
                    pg_avg = pg_df.get('validation_completion_pct', pd.Series(dtype=float)).mean()
                    
                    print(f"{product_group}:")
                    print(f"  📊 Records: {len(pg_df)}")
                    print(f"  ✅ Complete: {pg_complete} ({pg_complete/len(pg_df)*100:.1f}%)")
                    print(f"  📈 Avg completion: {pg_avg:.1f}%")
                    print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract network configuration data by product group')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output file path', required=True)
    parser.add_argument('--excel', help='Export to Excel format', action='store_true')
    parser.add_argument('-s', '--summary', help='Show detailed summary', action='store_true')
    
    args = parser.parse_args()
    
    # Validate output format
    if args.excel and not args.output.endswith(('.xlsx', '.xls')):
        args.output += '.xlsx'
    elif not args.excel and not args.output.endswith('.csv'):
        args.output += '.csv'
    
    # Process the file
    extractor = NetworkConfigExtractor()
    df = extractor.process_csv(args.input_file, args.output, args.excel)
    
    if df is not None:
        print("🎉 Extraction completed successfully!")
        
        if args.summary:
            extractor.print_summary(df)
        
        print(f"📁 Output saved to: {args.output}")
        print(f"📊 Total columns: {len(df.columns)}")
    else:
        print("❌ Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()