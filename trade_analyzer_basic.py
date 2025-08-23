#!/usr/bin/env python3
"""
Ross Cameron Trade Analysis - Basic Version
Uses only built-in Python libraries + pandas for initial analysis

Run this first, then install advanced libraries for full features:
pip install plotly backtrader yfinance TA-Lib matplotlib seaborn
"""

import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicTradeAnalyzer:
    """
    Basic trade analysis using only standard libraries
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def parse_integration_test_log(self, log_output: str) -> List[Dict]:
        """
        Parse integration test log output to extract trade data
        """
        trades = []
        lines = log_output.split('\n')
        
        current_trade = {}
        
        for line in lines:
            # Look for entry confirmations
            if "ENTERED" in line and "shares at" in line:
                # Parse: "ENTERED LGHL: 1874 shares at $2.40"
                parts = line.split("ENTERED ")[1].split(":")
                symbol = parts[0].strip()
                share_info = parts[1].strip()
                
                # Extract shares and price
                shares = int(share_info.split(" shares at $")[0])
                price = float(share_info.split("$")[1])
                
                trades.append({
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': price,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
        
        return trades
    
    def analyze_trades_from_log(self, log_file_path: str = None) -> pd.DataFrame:
        """
        Analyze trades from integration test log
        """
        
        # Sample data from our test run
        trades_data = [
            {
                'symbol': 'LGHL',
                'entry_time': '06:44',
                'entry_price': 2.40,
                'shares': 1874,
                'risk_amount': 375,
                'stop_price': 2.20,
                'target_price': 2.80,
                'strategy': 'STRONG_SQUEEZE_HIGH_RVOL',
                'rvol': 1649
            },
            {
                'symbol': 'SUUN',
                'entry_time': '06:58', 
                'entry_price': 1.69,
                'shares': 1875,
                'risk_amount': 375,
                'stop_price': 1.49,
                'target_price': 2.09,
                'strategy': 'STRONG_SQUEEZE_HIGH_RVOL',
                'rvol': 534
            },
            {
                'symbol': 'BSLK',
                'entry_time': '07:00',
                'entry_price': 3.94,
                'shares': 1874, 
                'risk_amount': 375,
                'stop_price': 3.74,
                'target_price': 4.34,
                'strategy': 'STRONG_SQUEEZE_HIGH_RVOL',
                'rvol': 1961
            }
        ]
        
        # Create DataFrame
        df = pd.DataFrame(trades_data)
        
        # Calculate derived metrics
        df['position_value'] = df['shares'] * df['entry_price']
        df['risk_percent'] = (df['entry_price'] - df['stop_price']) / df['entry_price'] * 100
        df['reward_percent'] = (df['target_price'] - df['entry_price']) / df['entry_price'] * 100
        df['risk_reward_ratio'] = df['reward_percent'] / df['risk_percent']
        
        # Ross Cameron Quality Scores
        df['rvol_score'] = df['rvol'].apply(self._score_rvol)
        df['risk_mgmt_score'] = df['risk_percent'].apply(self._score_risk_management)
        df['rr_score'] = df['risk_reward_ratio'].apply(self._score_risk_reward)
        
        # Overall entry quality (0-100)
        df['entry_quality'] = (df['rvol_score'] + df['risk_mgmt_score'] + df['rr_score']) / 3
        
        return df
    
    def _score_rvol(self, rvol: float) -> float:
        """Score RVOL strength (0-100)"""
        if rvol >= 5000:
            return 100
        elif rvol >= 2000:
            return 85
        elif rvol >= 1000:
            return 70
        elif rvol >= 500:
            return 55
        else:
            return 30
    
    def _score_risk_management(self, risk_percent: float) -> float:
        """Score risk management (0-100)"""
        if 1.5 <= risk_percent <= 2.5:
            return 100  # Optimal 2% risk
        elif 1.0 <= risk_percent <= 3.0:
            return 80   # Acceptable range
        elif risk_percent <= 4.0:
            return 60   # Too wide
        else:
            return 20   # Poor risk management
    
    def _score_risk_reward(self, rr_ratio: float) -> float:
        """Score risk/reward ratio (0-100)"""
        if rr_ratio >= 3.0:
            return 100
        elif rr_ratio >= 2.0:
            return 85   # Ross Cameron target
        elif rr_ratio >= 1.5:
            return 70
        elif rr_ratio >= 1.0:
            return 40
        else:
            return 10
    
    def generate_analysis_report(self, trades_df: pd.DataFrame) -> str:
        """Generate comprehensive text report"""
        
        report = f"""
ROSS CAMERON STRATEGY - TRADE ANALYSIS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
{'='*30}
Total Trades: {len(trades_df)}
Total Capital Deployed: ${trades_df['position_value'].sum():,.0f}
Total Risk Amount: ${trades_df['risk_amount'].sum():,.0f}
Average Entry Quality: {trades_df['entry_quality'].mean():.1f}/100

TRADE DETAILS
{'='*30}"""
        
        for idx, trade in trades_df.iterrows():
            report += f"""

Trade #{idx + 1}: {trade['symbol']}
  Entry Time: {trade['entry_time']}
  Entry Price: ${trade['entry_price']:.2f}
  Position Size: {trade['shares']:,} shares (${trade['position_value']:,.0f})
  Stop Loss: ${trade['stop_price']:.2f} ({trade['risk_percent']:.1f}% risk)
  Target: ${trade['target_price']:.2f} ({trade['reward_percent']:.1f}% reward)
  Risk/Reward: {trade['risk_reward_ratio']:.2f}:1
  RVOL: {trade['rvol']:,}%
  Entry Quality: {trade['entry_quality']:.1f}/100
  
  Quality Breakdown:
    - RVOL Strength: {trade['rvol_score']:.0f}/100 ({self._interpret_rvol(trade['rvol'])})
    - Risk Management: {trade['risk_mgmt_score']:.0f}/100 ({self._interpret_risk(trade['risk_percent'])})
    - Risk/Reward: {trade['rr_score']:.0f}/100 ({self._interpret_rr(trade['risk_reward_ratio'])})"""
        
        # Performance Analysis
        excellent_trades = len(trades_df[trades_df['entry_quality'] >= 80])
        good_trades = len(trades_df[trades_df['entry_quality'] >= 70])
        
        report += f"""

PERFORMANCE ANALYSIS
{'='*30}
Entry Quality Distribution:
  - Excellent (80-100): {excellent_trades} trades ({excellent_trades/len(trades_df)*100:.1f}%)
  - Good (70-79): {good_trades - excellent_trades} trades
  - Fair (60-69): {len(trades_df[(trades_df['entry_quality'] >= 60) & (trades_df['entry_quality'] < 70)])} trades
  - Poor (<60): {len(trades_df[trades_df['entry_quality'] < 60])} trades

Risk Management Analysis:
  - Average Risk per Trade: {trades_df['risk_percent'].mean():.1f}%
  - Optimal Risk (1.5-2.5%): {len(trades_df[(trades_df['risk_percent'] >= 1.5) & (trades_df['risk_percent'] <= 2.5)])} trades
  - Risk Budget Used: ${trades_df['risk_amount'].sum()} / $1,500 ({trades_df['risk_amount'].sum()/1500*100:.1f}%)

RVOL Analysis:
  - Average RVOL: {trades_df['rvol'].mean():,.0f}%
  - Highest RVOL: {trades_df['rvol'].max():,}% ({trades_df.loc[trades_df['rvol'].idxmax(), 'symbol']})
  - ACTION Quality: {len(trades_df[trades_df['rvol'] >= 1000])} trades with >1000% RVOL

RECOMMENDATIONS
{'='*30}"""
        
        # Generate recommendations
        if trades_df['entry_quality'].mean() >= 80:
            report += "\n✅ EXCELLENT: Strategy execution is on target"
        elif trades_df['entry_quality'].mean() >= 70:
            report += "\n🟡 GOOD: Minor improvements needed"
        else:
            report += "\n🔴 NEEDS WORK: Review entry criteria"
        
        if trades_df['risk_percent'].mean() > 2.5:
            report += "\n⚠️  Risk per trade too high - tighten stops"
        
        if trades_df['risk_reward_ratio'].mean() < 2.0:
            report += "\n⚠️  Risk/reward below Ross Cameron target of 2:1"
        
        low_rvol_trades = len(trades_df[trades_df['rvol'] < 1000])
        if low_rvol_trades > 0:
            report += f"\n⚠️  {low_rvol_trades} trades with low RVOL (<1000%) - filter more strictly"
        
        return report
    
    def _interpret_rvol(self, rvol: float) -> str:
        """Interpret RVOL strength"""
        if rvol >= 5000:
            return "EXPLOSIVE"
        elif rvol >= 2000:
            return "VERY STRONG"
        elif rvol >= 1000:
            return "STRONG"
        elif rvol >= 500:
            return "ACCEPTABLE"
        else:
            return "WEAK"
    
    def _interpret_risk(self, risk_pct: float) -> str:
        """Interpret risk management"""
        if 1.5 <= risk_pct <= 2.5:
            return "OPTIMAL"
        elif risk_pct < 1.5:
            return "TOO TIGHT"
        elif risk_pct <= 3.0:
            return "ACCEPTABLE"
        else:
            return "TOO WIDE"
    
    def _interpret_rr(self, rr: float) -> str:
        """Interpret risk/reward ratio"""
        if rr >= 3.0:
            return "EXCELLENT"
        elif rr >= 2.0:
            return "TARGET MET"
        elif rr >= 1.5:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def save_analysis(self, trades_df: pd.DataFrame, report: str):
        """Save analysis to files"""
        
        # Save CSV data
        csv_file = self.results_dir / 'trade_analysis.csv'
        trades_df.to_csv(csv_file, index=False)
        
        # Save report
        report_file = self.results_dir / 'trade_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis saved to {self.results_dir}/")
        logger.info(f"  - {csv_file.name} (trade data)")
        logger.info(f"  - {report_file.name} (analysis report)")
        
        return csv_file, report_file

def main():
    """Run basic trade analysis"""
    
    print("🔍 ROSS CAMERON TRADE ANALYZER (Basic Version)")
    print("="*60)
    
    # Create analyzer
    analyzer = BasicTradeAnalyzer()
    
    # Analyze trades
    print("📊 Analyzing trades from integration test...")
    trades_df = analyzer.analyze_trades_from_log()
    
    # Generate report
    print("📝 Generating analysis report...")
    report = analyzer.generate_analysis_report(trades_df)
    
    # Save results
    csv_file, report_file = analyzer.save_analysis(trades_df, report)
    
    # Display summary
    print("\n" + "="*60)
    print("📊 ANALYSIS COMPLETE")
    print("="*60)
    print(f"Trades Analyzed: {len(trades_df)}")
    print(f"Average Entry Quality: {trades_df['entry_quality'].mean():.1f}/100")
    print(f"Best Trade: {trades_df.loc[trades_df['entry_quality'].idxmax(), 'symbol']} ({trades_df['entry_quality'].max():.1f}/100)")
    print(f"Files saved: {csv_file.name}, {report_file.name}")
    
    # Print key findings
    print(f"\n🎯 KEY FINDINGS:")
    print(f"  - Total Risk: ${trades_df['risk_amount'].sum()} (Budget: $1,500)")
    print(f"  - Average RVOL: {trades_df['rvol'].mean():,.0f}%")
    print(f"  - Average R/R: {trades_df['risk_reward_ratio'].mean():.2f}:1")
    
    print(f"\n📁 Next Steps:")
    print(f"  - Review {report_file}")
    print(f"  - Install advanced libraries: pip install plotly backtrader")
    print(f"  - Run full analyzer: python trade_analyzer.py")
    
    return trades_df

if __name__ == "__main__":
    main()