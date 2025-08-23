#!/usr/bin/env python3
"""
Data Inspector - Examine ACTION alerts and raw market data
Helps validate if alerts are legitimate and understand the data
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataInspector:
    """
    Inspect HOD momentum alerts and underlying market data
    """
    
    def __init__(self):
        self.scan_data = None
        self.alerts = []
        
    def load_hod_scan_data(self, scan_file: str) -> Dict:
        """Load HOD momentum scan data"""
        try:
            with open(scan_file, 'r') as f:
                self.scan_data = json.load(f)
            
            logger.info(f"Loaded scan data for {self.scan_data.get('test_date', 'unknown date')}")
            logger.info(f"Total alerts: {self.scan_data.get('total_alerts', 0)}")
            logger.info(f"High priority alerts: {self.scan_data.get('high_priority_alerts', 0)}")
            
            return self.scan_data
            
        except Exception as e:
            logger.error(f"Error loading scan data: {e}")
            return {}
    
    def find_symbol_alerts(self, symbol: str) -> List[Dict]:
        """Find all alerts for a specific symbol"""
        if not self.scan_data:
            logger.error("No scan data loaded")
            return []
        
        symbol_alerts = []
        
        for alert in self.scan_data.get('all_alerts', []):
            if alert.get('symbol') == symbol:
                symbol_alerts.append(alert)
        
        return symbol_alerts
    
    def analyze_symbol_progression(self, symbol: str) -> pd.DataFrame:
        """Analyze how a symbol's alerts progressed throughout the day"""
        
        alerts = self.find_symbol_alerts(symbol)
        
        if not alerts:
            logger.warning(f"No alerts found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(alerts)
        
        # Sort by time
        df['time_sort'] = pd.to_datetime(df['time'], format='%H:%M')
        df = df.sort_values('time_sort')
        
        # Add progression analysis
        df['alert_sequence'] = range(1, len(df) + 1)
        df['price_change'] = df['price'].pct_change() * 100
        df['rvol_change'] = df['rvol_5min'].pct_change() * 100
        
        return df
    
    def get_action_alerts_summary(self) -> pd.DataFrame:
        """Get summary of all ACTION alerts"""
        if not self.scan_data:
            return pd.DataFrame()
        
        action_alerts = []
        
        for alert in self.scan_data.get('all_alerts', []):
            if alert.get('priority') == 'ACTION':
                action_alerts.append(alert)
        
        if not action_alerts:
            logger.warning("No ACTION alerts found")
            return pd.DataFrame()
        
        df = pd.DataFrame(action_alerts)
        
        # Sort by time
        df['time_sort'] = pd.to_datetime(df['time'], format='%H:%M')
        df = df.sort_values('time_sort')
        
        return df
    
    def examine_lghl_specifically(self) -> Dict:
        """Deep dive into LGHL alerts"""
        
        logger.info("🔍 Examining LGHL alerts specifically...")
        
        lghl_alerts = self.find_symbol_alerts('LGHL')
        
        if not lghl_alerts:
            logger.warning("❌ No LGHL alerts found!")
            return {'found': False, 'reason': 'No alerts in data'}
        
        logger.info(f"📊 Found {len(lghl_alerts)} LGHL alerts")
        
        # Analyze progression
        df = self.analyze_symbol_progression('LGHL')
        
        # Find ACTION alerts specifically
        action_alerts = [alert for alert in lghl_alerts if alert.get('priority') == 'ACTION']
        
        analysis = {
            'found': True,
            'total_alerts': len(lghl_alerts),
            'action_alerts': len(action_alerts),
            'first_alert_time': lghl_alerts[0].get('time') if lghl_alerts else None,
            'first_action_time': action_alerts[0].get('time') if action_alerts else None,
            'progression': df.to_dict('records') if not df.empty else [],
            'action_details': action_alerts
        }
        
        return analysis
    
    def print_symbol_timeline(self, symbol: str):
        """Print detailed timeline for a symbol"""
        
        print(f"\n{'='*60}")
        print(f"📊 DETAILED TIMELINE FOR {symbol}")
        print(f"{'='*60}")
        
        df = self.analyze_symbol_progression(symbol)
        
        if df.empty:
            print(f"❌ No alerts found for {symbol}")
            return
        
        print(f"Total alerts: {len(df)}")
        print(f"Time range: {df['time'].iloc[0]} - {df['time'].iloc[-1]}")
        
        # Show each alert
        for idx, alert in df.iterrows():
            priority_emoji = "🚨" if alert['priority'] == 'ACTION' else "⚠️" if alert['priority'] == 'HIGH' else "ℹ️"
            
            print(f"\n{priority_emoji} Alert #{alert['alert_sequence']} - {alert['time']}")
            print(f"   Priority: {alert['priority']}")
            print(f"   Strategy: {alert['strategy']}")
            print(f"   Price: ${alert['price']:.2f} ({alert.get('price_change', 0):+.1f}%)")
            print(f"   RVOL: {alert['rvol_5min']:,.0f}% ({alert.get('rvol_change', 0):+.1f}%)")
            print(f"   Description: {alert.get('description', 'N/A')}")
            
            if alert['priority'] == 'ACTION':
                print(f"   🎯 THIS IS AN ACTION ALERT!")
                print(f"   Float: {alert.get('float', 'Unknown')}")
                print(f"   Low Float: {alert.get('is_low_float', 'Unknown')}")
                print(f"   Former Momo: {alert.get('former_momo', 'Unknown')}")
    
    def validate_action_criteria(self, alert: Dict) -> Dict:
        """Validate if an alert meets ACTION criteria"""
        
        validation = {
            'meets_criteria': True,
            'issues': []
        }
        
        # Check RVOL threshold (should be >500% for ACTION)
        rvol = alert.get('rvol_5min', 0)
        if rvol < 500:
            validation['meets_criteria'] = False
            validation['issues'].append(f"RVOL too low: {rvol:.0f}% (need >500%)")
        
        # Check strategy
        strategy = alert.get('strategy', '')
        if strategy != 'STRONG_SQUEEZE_HIGH_RVOL':
            validation['meets_criteria'] = False
            validation['issues'].append(f"Wrong strategy: {strategy}")
        
        # Check price category
        price = alert.get('price', 0)
        if price < 1 or price > 20:
            validation['issues'].append(f"Price outside $1-$20 range: ${price:.2f}")
        
        return validation
    
    def create_alerts_summary_report(self) -> str:
        """Create comprehensive alerts summary"""
        
        if not self.scan_data:
            return "No scan data loaded"
        
        # Get ACTION alerts
        action_df = self.get_action_alerts_summary()
        
        report = f"""
ACTION ALERTS ANALYSIS REPORT
{'='*50}
Date: {self.scan_data.get('test_date', 'Unknown')}
Time Range: {self.scan_data.get('time_range', 'Unknown')}

SUMMARY STATISTICS
{'-'*30}
Total Alerts: {self.scan_data.get('total_alerts', 0):,}
High Priority: {self.scan_data.get('high_priority_alerts', 0):,}
ACTION Alerts: {len(action_df)}

ACTION ALERTS BREAKDOWN
{'-'*30}"""

        if not action_df.empty:
            # Group by symbol
            symbol_counts = action_df['symbol'].value_counts()
            
            report += f"\nMost Active Symbols:"
            for symbol, count in symbol_counts.head(10).items():
                first_time = action_df[action_df['symbol'] == symbol]['time'].iloc[0]
                avg_rvol = action_df[action_df['symbol'] == symbol]['rvol_5min'].mean()
                report += f"\n  {symbol}: {count} alerts (first: {first_time}, avg RVOL: {avg_rvol:,.0f}%)"
            
            # Time distribution
            report += f"\n\nTime Distribution:"
            time_dist = action_df['time'].str[:2].value_counts().sort_index()
            for hour, count in time_dist.items():
                report += f"\n  {hour}:xx - {count} alerts"
            
            # RVOL statistics
            report += f"\n\nRVOL Analysis:"
            report += f"\n  Average RVOL: {action_df['rvol_5min'].mean():,.0f}%"
            report += f"\n  Median RVOL: {action_df['rvol_5min'].median():,.0f}%"
            report += f"\n  Max RVOL: {action_df['rvol_5min'].max():,.0f}% ({action_df.loc[action_df['rvol_5min'].idxmax(), 'symbol']})"
            
        return report

def main():
    """Main inspection function"""
    
    print("🔍 DATA INSPECTOR - Examining ACTION Alerts")
    print("="*60)
    
    # Initialize inspector
    inspector = DataInspector()
    
    # Load scan data
    scan_file = "/Users/claytonsmacbookpro/Projects/warrior_bt/results/hod_momentum_scans/hod_momentum_scan_2025-08-13.json"
    
    print(f"📂 Loading scan data from: {Path(scan_file).name}")
    scan_data = inspector.load_hod_scan_data(scan_file)
    
    if not scan_data:
        print("❌ Failed to load scan data")
        return
    
    # Examine LGHL specifically
    print(f"\n🎯 EXAMINING LGHL ALERTS...")
    lghl_analysis = inspector.examine_lghl_specifically()
    
    if lghl_analysis['found']:
        print(f"✅ LGHL alerts found!")
        print(f"   Total alerts: {lghl_analysis['total_alerts']}")
        print(f"   ACTION alerts: {lghl_analysis['action_alerts']}")
        
        if lghl_analysis['action_alerts'] > 0:
            print(f"   First ACTION alert: {lghl_analysis['first_action_time']}")
            
            # Show detailed timeline
            inspector.print_symbol_timeline('LGHL')
            
            # Validate first ACTION alert
            first_action = lghl_analysis['action_details'][0]
            validation = inspector.validate_action_criteria(first_action)
            
            print(f"\n🔍 ACTION ALERT VALIDATION:")
            print(f"   Meets criteria: {'✅ YES' if validation['meets_criteria'] else '❌ NO'}")
            if validation['issues']:
                print(f"   Issues found:")
                for issue in validation['issues']:
                    print(f"     - {issue}")
        else:
            print(f"   ❌ No ACTION alerts found for LGHL")
    else:
        print(f"❌ No LGHL alerts found: {lghl_analysis.get('reason', 'Unknown')}")
    
    # Create summary report
    print(f"\n📊 GENERATING SUMMARY REPORT...")
    report = inspector.create_alerts_summary_report()
    
    # Save report
    report_file = Path("results/alerts_inspection_report.txt")
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"💾 Report saved to: {report_file}")
    
    # Show key findings
    action_df = inspector.get_action_alerts_summary()
    if not action_df.empty:
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   Total ACTION alerts: {len(action_df)}")
        print(f"   First ACTION alert: {action_df['time'].iloc[0]} ({action_df['symbol'].iloc[0]})")
        print(f"   Highest RVOL: {action_df['rvol_5min'].max():,.0f}% ({action_df.loc[action_df['rvol_5min'].idxmax(), 'symbol']})")
        
        # Check if LGHL is in ACTION alerts
        lghl_actions = action_df[action_df['symbol'] == 'LGHL']
        if not lghl_actions.empty:
            print(f"   ✅ LGHL ACTION alerts: {len(lghl_actions)} found")
            print(f"   First LGHL ACTION: {lghl_actions['time'].iloc[0]}")
        else:
            print(f"   ❌ LGHL not found in ACTION alerts")

if __name__ == "__main__":
    main()