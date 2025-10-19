# output_manager.py - نظام إدارة المخرجات المحسن
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any
import logging

class OutputManager:
    def __init__(self, base_path="output"):
        self.base_path = base_path
        self.backtest_base_path = "backtest"
        self.setup_directories()
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """إنشاء هيكل المجلدات المنظم"""
        directories = [
            'training',
            'predictions', 
            'betting_recommendations',
            'performance_analysis',
            'weekly_reports',
            'visualizations/charts',
            'visualizations/heatmaps',
            'data/exports',
            'models/saved_weights',
            # مجلدات Backtest الجديدة
            'backtest/results',
            'backtest/plots',
            'backtest/csv_data',
            'backtest/betting_analysis',
            'backtest/performance_reports',
            'backtest/season_analysis',
            'backtest/comparisons',
            'backtest/execution_logs'
        ]
        
        for directory in directories:
            os.makedirs(f"{self.base_path}/{directory}", exist_ok=True)
    
    def _save_json(self, data: Dict, filename: str) -> str:
        """حفظ البيانات كملف JSON"""
        try:
            # التأكد من وجود المجلد
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 تم حفظ JSON: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ JSON {filename}: {e}")
            return None
    
    def _save_csv(self, data: Any, filename: str) -> str:
        """حفظ البيانات كملف CSV"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                data.to_csv(filename, index=False, encoding='utf-8')
            elif isinstance(data, list):
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding='utf-8')
            elif isinstance(data, dict):
                # إذا كان dict بسيط، حفظ كصف واحد
                df = pd.DataFrame([data])
                df.to_csv(filename, index=False, encoding='utf-8')
            else:
                self.logger.warning(f"⚠️ نوع بيانات غير معروف لـ CSV: {type(data)}")
                return None
            
            self.logger.info(f"💾 تم حفظ CSV: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ CSV {filename}: {e}")
            return None

    def save_comprehensive_backtest_results(self, backtest_results: Dict, backtest_id: str = None) -> Dict:
        """حفظ نتائج Backtest الشاملة بجميع الأشكال"""
        
        if not backtest_id:
            backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        saved_files = {
            'backtest_id': backtest_id,
            'json_files': [],
            'csv_files': [],
            'plot_files': [],
            'report_files': []
        }
        
        try:
            # 1. حفظ النتائج الرئيسية كـ JSON
            main_results_file = f"{self.base_path}/backtest/results/comprehensive_backtest_{backtest_id}.json"
            saved_files['json_files'].append(self._save_json(backtest_results, main_results_file))
            
            # 2. حفظ البيانات الرئيسية كـ CSV
            csv_data = self._extract_backtest_csv_data(backtest_results)
            for data_name, data in csv_data.items():
                csv_file = f"{self.base_path}/backtest/csv_data/{data_name}_{backtest_id}.csv"
                saved_files['csv_files'].append(self._save_csv(data, csv_file))
            
            # 3. إنشاء المخططات والرسوم البيانية
            plot_files = self._create_backtest_plots(backtest_results, backtest_id)
            saved_files['plot_files'].extend(plot_files)
            
            # 4. إنشاء التقارير المفصلة
            report_files = self._generate_backtest_reports(backtest_results, backtest_id)
            saved_files['report_files'].extend(report_files)
            
            # 5. حفظ تحليل الرهان
            betting_files = self._save_betting_analysis(backtest_results, backtest_id)
            saved_files['json_files'].extend(betting_files)
            
            # 6. حفظ تحليل الموسم
            seasonal_files = self._save_seasonal_analysis(backtest_results, backtest_id)
            saved_files['json_files'].extend(seasonal_files)
            
            # 7. إنشاء تقرير تنفيذي
            executive_report = self._generate_executive_report(backtest_results, backtest_id, saved_files)
            saved_files['report_files'].append(executive_report)
            
            self.logger.info(f"✅ تم حفظ نتائج Backtest الشاملة: {len(saved_files['json_files'])} ملف JSON, "
                           f"{len(saved_files['csv_files'])} ملف CSV, {len(saved_files['plot_files'])} مخطط")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ نتائج Backtest الشاملة: {e}")
            return saved_files

    def _extract_backtest_csv_data(self, backtest_results: Dict) -> Dict[str, pd.DataFrame]:
        """استخراج البيانات لتحويلها إلى CSV"""
        csv_data = {}
        
        try:
            # بيانات الأداء الشامل
            overall_perf = backtest_results.get('overall_performance', {})
            if overall_perf:
                csv_data['overall_performance'] = pd.DataFrame([overall_perf])
            
            # بيانات الرهان
            betting_results = []
            seasonal_results = backtest_results.get('seasonal_results', {})
            
            for season_type, season_data in seasonal_results.items():
                if 'betting_results' in season_data:
                    for bet in season_data['betting_results']:
                        bet_record = {
                            'season_type': season_type,
                            'match': f"{bet['match_info']['home_team']} vs {bet['match_info']['away_team']}",
                            'date': bet['match_info'].get('date', ''),
                            'betting_decision': bet['betting_decision'].get('market', '') + '_' + bet['betting_decision'].get('selection', ''),
                            'stake': bet.get('stake', 0),
                            'odds': bet.get('odds', 0),
                            'profit': bet.get('profit', 0),
                            'net_profit': bet.get('net_profit', 0),
                            'result': bet.get('result', ''),
                            'bankroll_after': bet.get('bankroll_after', 0)
                        }
                        betting_results.append(bet_record)
            
            if betting_results:
                csv_data['betting_results'] = pd.DataFrame(betting_results)
            
            # تطور الرصيد
            bankroll_evolution = []
            for season_type, season_data in seasonal_results.items():
                if 'bankroll_evolution' in season_data:
                    for point in season_data['bankroll_evolution']:
                        point['season_type'] = season_type
                        bankroll_evolution.append(point)
            
            if bankroll_evolution:
                csv_data['bankroll_evolution'] = pd.DataFrame(bankroll_evolution)
            
            # أداء النماذج
            model_performance = backtest_results.get('model_performance', {})
            if model_performance:
                accuracy_data = []
                accuracy_analysis = model_performance.get('accuracy_analysis', {})
                for season_type, acc in accuracy_analysis.items():
                    acc_record = {
                        'season_type': season_type,
                        'score_accuracy': acc.get('score_accuracy', 0),
                        'result_accuracy': acc.get('result_accuracy', 0)
                    }
                    accuracy_data.append(acc_record)
                
                if accuracy_data:
                    csv_data['model_accuracy'] = pd.DataFrame(accuracy_data)
            
            return csv_data
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في استخراج بيانات CSV: {e}")
            return csv_data

    def _create_backtest_plots(self, backtest_results: Dict, backtest_id: str) -> List[str]:
        """إنشاء مخططات ورسوم بيانية لـ Backtest"""
        plot_files = []
        
        try:
            # 1. مخطط تطور الرصيد
            plot_files.append(self._create_bankroll_evolution_plot(backtest_results, backtest_id))
            
            # 2. مخطط أداء الرهان حسب الموسم
            plot_files.append(self._create_seasonal_performance_plot(backtest_results, backtest_id))
            
            # 3. مخطط توزيع الأرباح والخسائر
            plot_files.append(self._create_profit_distribution_plot(backtest_results, backtest_id))
            
            # 4. مخطط دقة النماذج
            plot_files.append(self._create_model_accuracy_plot(backtest_results, backtest_id))
            
            # 5. مخطط تحليل المخاطر
            plot_files.append(self._create_risk_analysis_plot(backtest_results, backtest_id))
            
            # 6. مخطط مقارنة الأسواق
            plot_files.append(self._create_market_comparison_plot(backtest_results, backtest_id))
            
            # 7. مخطط heatmap للأداء
            plot_files.append(self._create_performance_heatmap(backtest_results, backtest_id))
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء المخططات: {e}")
        
        return [f for f in plot_files if f is not None]

    def _create_bankroll_evolution_plot(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط تطور الرصيد عبر الزمن"""
        try:
            plt.figure(figsize=(15, 8))
            
            seasonal_results = backtest_results.get('seasonal_results', {})
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            
            for i, (season_type, season_data) in enumerate(seasonal_results.items()):
                if 'bankroll_evolution' in season_data and season_data['bankroll_evolution']:
                    evolution = season_data['bankroll_evolution']
                    matches = [point['match_count'] for point in evolution]
                    bankrolls = [point['bankroll'] for point in evolution]
                    
                    plt.plot(matches, bankrolls, label=season_type, 
                            color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3)
            
            plt.title('تطور رأس المال عبر المواسم', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('عدد المباريات', fontsize=12)
            plt.ylabel('رأس المال (£)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # إضافة خط رأس المال الابتدائي
            initial_bankroll = backtest_results.get('parameters', {}).get('initial_bankroll', 10000)
            plt.axhline(y=initial_bankroll, color='red', linestyle='--', alpha=0.7, label='رأس المال الابتدائي')
            
            plot_file = f"{self.base_path}/backtest/plots/bankroll_evolution_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط تطور الرصيد: {e}")
            return None

    def _create_seasonal_performance_plot(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط مقارنة أداء المواسم"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            seasonal_results = backtest_results.get('seasonal_results', {})
            seasons = []
            roi_values = []
            win_rates = []
            accuracy_values = []
            total_bets = []
            
            for season_type, season_data in seasonal_results.items():
                if 'performance_metrics' in season_data:
                    metrics = season_data['performance_metrics']
                    betting_perf = metrics.get('betting_performance', {})
                    model_acc = metrics.get('model_accuracy', {})
                    
                    seasons.append(season_type)
                    roi_values.append(betting_perf.get('roi', 0))
                    win_rates.append(betting_perf.get('win_rate', 0) * 100)
                    accuracy_values.append(model_acc.get('result_accuracy', 0) * 100)
                    total_bets.append(betting_perf.get('total_bets', 0))
            
            # مخطط ROI
            bars1 = ax1.bar(seasons, roi_values, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax1.set_title('العائد على الاستثمار (ROI) حسب الموسم', fontweight='bold')
            ax1.set_ylabel('ROI (%)')
            ax1.bar_label(bars1, fmt='%.1f%%', padding=3)
            
            # مخطط معدل الفوز
            bars2 = ax2.bar(seasons, win_rates, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax2.set_title('معدل الفوز حسب الموسم', fontweight='bold')
            ax2.set_ylabel('معدل الفوز (%)')
            ax2.bar_label(bars2, fmt='%.1f%%', padding=3)
            
            # مخطط الدقة
            bars3 = ax3.bar(seasons, accuracy_values, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax3.set_title('دقة النموذج حسب الموسم', fontweight='bold')
            ax3.set_ylabel('الدقة (%)')
            ax3.bar_label(bars3, fmt='%.1f%%', padding=3)
            
            # مخطط عدد الرهانات
            bars4 = ax4.bar(seasons, total_bets, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax4.set_title('إجمالي الرهانات حسب الموسم', fontweight='bold')
            ax4.set_ylabel('عدد الرهانات')
            ax4.bar_label(bars4, fmt='%d', padding=3)
            
            plt.tight_layout(pad=3.0)
            plot_file = f"{self.base_path}/backtest/plots/seasonal_performance_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط أداء المواسم: {e}")
            return None

    def _create_profit_distribution_plot(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط توزيع الأرباح والخسائر"""
        try:
            # جمع جميع نتائج الرهان
            all_betting_results = []
            seasonal_results = backtest_results.get('seasonal_results', {})
            
            for season_data in seasonal_results.values():
                if 'betting_results' in season_data:
                    all_betting_results.extend(season_data['betting_results'])
            
            if not all_betting_results:
                return None
            
            profits = [bet.get('profit', 0) for bet in all_betting_results]
            
            plt.figure(figsize=(15, 10))
            
            # توزيع الأرباح
            plt.subplot(2, 2, 1)
            plt.hist(profits, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.title('توزيع الأرباح والخسائر', fontweight='bold')
            plt.xlabel('الربح/الخسارة (£)')
            plt.ylabel('التكرار')
            
            # مخطط الصندوق
            plt.subplot(2, 2, 2)
            plt.boxplot(profits, vert=True)
            plt.title('مخطط الصندوق للأرباح', fontweight='bold')
            plt.ylabel('الربح/الخسارة (£)')
            
            # الرهانات الفائزة vs الخاسرة
            plt.subplot(2, 2, 3)
            winning_bets = [p for p in profits if p > 0]
            losing_bets = [p for p in profits if p < 0]
            push_bets = [p for p in profits if p == 0]
            
            categories = ['فائزة', 'خاسرة', 'متعادلة']
            counts = [len(winning_bets), len(losing_bets), len(push_bets)]
            colors = ['#28a745', '#dc3545', '#ffc107']
            
            bars = plt.bar(categories, counts, color=colors)
            plt.title('توزيع نتائج الرهان', fontweight='bold')
            plt.ylabel('عدد الرهانات')
            plt.bar_label(bars, fmt='%d', padding=3)
            
            # تحليل حجم الرهان vs الربح
            plt.subplot(2, 2, 4)
            stakes = [bet.get('stake', 0) for bet in all_betting_results]
            plt.scatter(stakes, profits, alpha=0.6, color='#6f42c1')
            plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
            plt.title('العلاقة بين حجم الرهان والربح', fontweight='bold')
            plt.xlabel('حجم الرهان (£)')
            plt.ylabel('الربح (£)')
            
            plt.tight_layout(pad=3.0)
            plot_file = f"{self.base_path}/backtest/plots/profit_distribution_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط توزيع الأرباح: {e}")
            return None

    def _create_model_accuracy_plot(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط دقة النماذج التنبؤية"""
        try:
            model_performance = backtest_results.get('model_performance', {})
            accuracy_analysis = model_performance.get('accuracy_analysis', {})
            
            if not accuracy_analysis:
                return None
            
            seasons = list(accuracy_analysis.keys())
            score_accuracies = [acc.get('score_accuracy', 0) * 100 for acc in accuracy_analysis.values()]
            result_accuracies = [acc.get('result_accuracy', 0) * 100 for acc in accuracy_analysis.values()]
            
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(seasons))
            width = 0.35
            
            plt.bar(x - width/2, score_accuracies, width, label='دقة النتيجة', color='#2E86AB')
            plt.bar(x + width/2, result_accuracies, width, label='دقة النتيجة (فوز/تعادل/خسارة)', color='#A23B72')
            
            plt.xlabel('نوع الموسم')
            plt.ylabel('الدقة (%)')
            plt.title('دقة النماذج التنبؤية عبر المواسم', fontweight='bold')
            plt.xticks(x, seasons)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # إضافة القيم على الأعمدة
            for i, (v1, v2) in enumerate(zip(score_accuracies, result_accuracies)):
                plt.text(i - width/2, v1 + 1, f'{v1:.1f}%', ha='center', va='bottom')
                plt.text(i + width/2, v2 + 1, f'{v2:.1f}%', ha='center', va='bottom')
            
            plot_file = f"{self.base_path}/backtest/plots/model_accuracy_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط دقة النماذج: {e}")
            return None

    def _create_risk_analysis_plot(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط تحليل المخاطر"""
        try:
            betting_analysis = backtest_results.get('betting_analysis', {})
            risk_analysis = betting_analysis.get('risk_analysis', {})
            
            if not risk_analysis:
                return None
            
            plt.figure(figsize=(15, 10))
            
            # مخطط التقلبية
            plt.subplot(2, 2, 1)
            volatility = risk_analysis.get('volatility', 0)
            plt.bar(['التقلبية'], [volatility], color='#ff6b6b')
            plt.title('تقلبية الأرباح', fontweight='bold')
            plt.ylabel('مقدار التقلبية')
            
            # مخطط أقصى انخفاض
            plt.subplot(2, 2, 2)
            max_drawdown = risk_analysis.get('drawdown_analysis', {}).get('max_drawdown', 0)
            plt.bar(['أقصى انخفاض'], [max_drawdown], color='#ee5a24')
            plt.title('أقصى انخفاض في رأس المال', fontweight='bold')
            plt.ylabel('النسبة المئوية (%)')
            
            # مخطط سلاسل الفوز والخسارة
            plt.subplot(2, 2, 3)
            drawdown_analysis = risk_analysis.get('drawdown_analysis', {})
            max_win_streak = drawdown_analysis.get('max_winning_streak', 0)
            max_loss_streak = drawdown_analysis.get('max_losing_streak', 0)
            
            categories = ['أطول سلسلة فوز', 'أطول سلسلة خسارة']
            values = [max_win_streak, max_loss_streak]
            colors = ['#28a745', '#dc3545']
            
            bars = plt.bar(categories, values, color=colors)
            plt.title('سلاسل الفوز والخسارة', fontweight='bold')
            plt.ylabel('عدد الرهانات المتتالية')
            plt.bar_label(bars, fmt='%d', padding=3)
            
            # مخطط عامل الربحية
            plt.subplot(2, 2, 4)
            profit_factor = risk_analysis.get('profit_factor', 0)
            plt.bar(['عامل الربحية'], [profit_factor], color='#10ac84')
            plt.title('عامل الربحية', fontweight='bold')
            plt.ylabel('النسبة')
            plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='نقطة التعادل')
            plt.legend()
            
            plt.tight_layout(pad=3.0)
            plot_file = f"{self.base_path}/backtest/plots/risk_analysis_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط تحليل المخاطر: {e}")
            return None

    def _create_market_comparison_plot(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط مقارنة أداء أسواق الرهان المختلفة"""
        try:
            betting_analysis = backtest_results.get('betting_analysis', {})
            by_market = betting_analysis.get('by_market', {})
            
            if not by_market:
                return None
            
            markets = list(by_market.keys())
            roi_values = [data.get('roi', 0) for data in by_market.values()]
            win_rates = [data.get('win_rate', 0) * 100 for data in by_market.values()]
            total_bets = [data.get('total_bets', 0) for data in by_market.values()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # مخطط ROI حسب السوق
            bars1 = ax1.bar(markets, roi_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax1.set_title('العائد على الاستثمار حسب سوق الرهان', fontweight='bold')
            ax1.set_ylabel('ROI (%)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.bar_label(bars1, fmt='%.1f%%', padding=3)
            
            # مخطط معدل الفوز حسب السوق
            bars2 = ax2.bar(markets, win_rates, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax2.set_title('معدل الفوز حسب سوق الرهان', fontweight='bold')
            ax2.set_ylabel('معدل الفوز (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.bar_label(bars2, fmt='%.1f%%', padding=3)
            
            plt.tight_layout(pad=3.0)
            plot_file = f"{self.base_path}/backtest/plots/market_comparison_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط مقارنة الأسواق: {e}")
            return None

    def _create_performance_heatmap(self, backtest_results: Dict, backtest_id: str) -> str:
        """مخطط heatmap للأداء الشامل"""
        try:
            # تجميع بيانات الأداء
            performance_data = []
            seasonal_results = backtest_results.get('seasonal_results', {})
            
            for season_type, season_data in seasonal_results.items():
                if 'performance_metrics' in season_data:
                    metrics = season_data['performance_metrics']
                    betting_perf = metrics.get('betting_performance', {})
                    
                    performance_data.append({
                        'الموسم': season_type,
                        'ROI': betting_perf.get('roi', 0),
                        'معدل الفوز': betting_perf.get('win_rate', 0) * 100,
                        'الدقة': metrics.get('model_accuracy', {}).get('result_accuracy', 0) * 100,
                        'الرهانات': betting_perf.get('total_bets', 0),
                        'الربح الإجمالي': betting_perf.get('total_profit', 0)
                    })
            
            if not performance_data:
                return None
            
            df = pd.DataFrame(performance_data)
            df.set_index('الموسم', inplace=True)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, linewidths=1, cbar_kws={'label': 'القيمة'})
            
            plt.title('خريطة حرارة لأداء المواسم', fontweight='bold', pad=20)
            
            plot_file = f"{self.base_path}/backtest/plots/performance_heatmap_{backtest_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_file
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء مخطط heatmap: {e}")
            return None

    def _generate_backtest_reports(self, backtest_results: Dict, backtest_id: str) -> List[str]:
        """إنشاء تقارير Backtest المفصلة"""
        report_files = []
        
        try:
            # تقرير الأداء الشامل
            performance_report = self._generate_performance_report(backtest_results, backtest_id)
            if performance_report:
                report_files.append(performance_report)
            
            # تقرير تحليل الرهان
            betting_report = self._generate_betting_report(backtest_results, backtest_id)
            if betting_report:
                report_files.append(betting_report)
            
            # تقرير تحليل النماذج
            model_report = self._generate_model_report(backtest_results, backtest_id)
            if model_report:
                report_files.append(model_report)
            
            # تقرير المخاطر
            risk_report = self._generate_risk_report(backtest_results, backtest_id)
            if risk_report:
                report_files.append(risk_report)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء التقارير: {e}")
        
        return report_files

    def _generate_performance_report(self, backtest_results: Dict, backtest_id: str) -> str:
        """تقرير الأداء الشامل"""
        try:
            overall_perf = backtest_results.get('overall_performance', {})
            
            report = {
                'report_type': 'performance_report',
                'backtest_id': backtest_id,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'الدقة_العامة': f"{overall_perf.get('overall_accuracy', 0) * 100:.1f}%",
                    'العائد_على_الاستثمار': f"{overall_perf.get('overall_roi', 0):.1f}%",
                    'معدل_الفوز': f"{overall_perf.get('overall_win_rate', 0) * 100:.1f}%",
                    'التقييم': overall_perf.get('performance_rating', 'غير معروف'),
                    'إجمالي_الرهانات': overall_perf.get('total_bets_placed', 0),
                    'إجمالي_التنبؤات': overall_perf.get('total_predictions_made', 0)
                },
                'key_metrics': overall_perf
            }
            
            report_file = f"{self.base_path}/backtest/performance_reports/performance_report_{backtest_id}.json"
            return self._save_json(report, report_file)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير الأداء: {e}")
            return None

    def _generate_betting_report(self, backtest_results: Dict, backtest_id: str) -> str:
        """تقرير تحليل الرهان"""
        try:
            betting_analysis = backtest_results.get('betting_analysis', {})
            
            report = {
                'report_type': 'betting_analysis_report',
                'backtest_id': backtest_id,
                'timestamp': datetime.now().isoformat(),
                'market_performance': betting_analysis.get('by_market', {}),
                'seasonal_performance': betting_analysis.get('by_season', {}),
                'risk_analysis': betting_analysis.get('risk_analysis', {}),
                'recommendations': betting_analysis.get('recommendations', [])
            }
            
            report_file = f"{self.base_path}/backtest/betting_analysis/betting_report_{backtest_id}.json"
            return self._save_json(report, report_file)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير الرهان: {e}")
            return None

    def _generate_model_report(self, backtest_results: Dict, backtest_id: str) -> str:
        """تقرير تحليل النماذج"""
        try:
            model_performance = backtest_results.get('model_performance', {})
            
            report = {
                'report_type': 'model_performance_report',
                'backtest_id': backtest_id,
                'timestamp': datetime.now().isoformat(),
                'accuracy_analysis': model_performance.get('accuracy_analysis', {}),
                'consistency_analysis': model_performance.get('consistency_analysis', {}),
                'improvement_recommendations': model_performance.get('improvement_recommendations', [])
            }
            
            report_file = f"{self.base_path}/backtest/performance_reports/model_report_{backtest_id}.json"
            return self._save_json(report, report_file)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير النماذج: {e}")
            return None

    def _generate_risk_report(self, backtest_results: Dict, backtest_id: str) -> str:
        """تقرير تحليل المخاطر"""
        try:
            betting_analysis = backtest_results.get('betting_analysis', {})
            risk_analysis = betting_analysis.get('risk_analysis', {})
            
            report = {
                'report_type': 'risk_analysis_report',
                'backtest_id': backtest_id,
                'timestamp': datetime.now().isoformat(),
                'volatility': risk_analysis.get('volatility', 0),
                'consistency': risk_analysis.get('consistency', 0),
                'drawdown_analysis': risk_analysis.get('drawdown_analysis', {}),
                'betting_patterns': risk_analysis.get('betting_patterns', {}),
                'risk_rating': self._calculate_risk_rating(risk_analysis)
            }
            
            report_file = f"{self.base_path}/backtest/performance_reports/risk_report_{backtest_id}.json"
            return self._save_json(report, report_file)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير المخاطر: {e}")
            return None

    def _calculate_risk_rating(self, risk_analysis: Dict) -> str:
        """تقييم مستوى المخاطرة"""
        try:
            volatility = risk_analysis.get('volatility', 0)
            max_drawdown = risk_analysis.get('drawdown_analysis', {}).get('max_drawdown', 0)
            
            if volatility < 100 and max_drawdown < 10:
                return "منخفض"
            elif volatility < 300 and max_drawdown < 25:
                return "متوسط"
            else:
                return "مرتفع"
        except:
            return "غير معروف"

    def _save_betting_analysis(self, backtest_results: Dict, backtest_id: str) -> List[str]:
        """حفظ تحليل الرهان"""
        saved_files = []
        
        try:
            betting_analysis = backtest_results.get('betting_analysis', {})
            if betting_analysis:
                betting_file = f"{self.base_path}/backtest/betting_analysis/betting_analysis_{backtest_id}.json"
                saved_files.append(self._save_json(betting_analysis, betting_file))
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ تحليل الرهان: {e}")
        
        return saved_files

    def _save_seasonal_analysis(self, backtest_results: Dict, backtest_id: str) -> List[str]:
        """حفظ تحليل الموسم"""
        saved_files = []
        
        try:
            seasonal_results = backtest_results.get('seasonal_results', {})
            if seasonal_results:
                seasonal_file = f"{self.base_path}/backtest/season_analysis/seasonal_analysis_{backtest_id}.json"
                saved_files.append(self._save_json(seasonal_results, seasonal_file))
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ تحليل الموسم: {e}")
        
        return saved_files

    def _generate_executive_report(self, backtest_results: Dict, backtest_id: str, saved_files: Dict) -> str:
        """تقرير تنفيذي شامل"""
        try:
            overall_perf = backtest_results.get('overall_performance', {})
            betting_analysis = backtest_results.get('betting_analysis', {})
            model_perf = backtest_results.get('model_performance', {})
            
            executive_report = {
                'report_type': 'executive_summary',
                'backtest_id': backtest_id,
                'timestamp': datetime.now().isoformat(),
                'overview': {
                    'فترة_التحليل': backtest_results.get('parameters', {}),
                    'مدة_التشغيل': f"{backtest_results.get('start_time', '')} إلى {backtest_results.get('end_time', '')}",
                    'نجاح_التشغيل': backtest_results.get('success', False)
                },
                'key_results': {
                    'الدقة_العامة': f"{overall_perf.get('overall_accuracy', 0) * 100:.1f}%",
                    'العائد_على_الاستثمار': f"{overall_perf.get('overall_roi', 0):.1f}%",
                    'معدل_الفوز': f"{overall_perf.get('overall_win_rate', 0) * 100:.1f}%",
                    'إجمالي_الرهانات': overall_perf.get('total_bets_placed', 0),
                    'التقييم_النهائي': overall_perf.get('performance_rating', 'غير معروف')
                },
                'strengths': self._identify_strengths(overall_perf, betting_analysis),
                'improvements': self._identify_improvements(overall_perf, betting_analysis),
                'recommendations': self._generate_final_recommendations(backtest_results),
                'saved_files_summary': {
                    'json_files': len(saved_files.get('json_files', [])),
                    'csv_files': len(saved_files.get('csv_files', [])),
                    'plot_files': len(saved_files.get('plot_files', [])),
                    'report_files': len(saved_files.get('report_files', []))
                }
            }
            
            report_file = f"{self.base_path}/backtest/performance_reports/executive_report_{backtest_id}.json"
            return self._save_json(executive_report, report_file)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء التقرير التنفيذي: {e}")
            return None

    def _identify_strengths(self, overall_perf: Dict, betting_analysis: Dict) -> List[str]:
        """تحديد نقاط القوة"""
        strengths = []
        
        try:
            if overall_perf.get('overall_roi', 0) > 10:
                strengths.append("ربحية عالية من الرهانات")
            
            if overall_perf.get('overall_accuracy', 0) > 0.5:
                strengths.append("دقة تنبؤ جيدة")
            
            market_perf = betting_analysis.get('by_market', {})
            for market, perf in market_perf.items():
                if perf.get('roi', 0) > 15:
                    strengths.append(f"أداء ممتاز في رهانات {market}")
            
            return strengths if strengths else ["أداء متوازن في معظم المجالات"]
            
        except Exception as e:
            return ["تحتاج إلى مزيد من التحليل"]

    def _identify_improvements(self, overall_perf: Dict, betting_analysis: Dict) -> List[str]:
        """تحديد مجالات التحسين"""
        improvements = []
        
        try:
            if overall_perf.get('overall_accuracy', 0) < 0.45:
                improvements.append("تحسين دقة النماذج التنبؤية")
            
            risk_analysis = betting_analysis.get('risk_analysis', {})
            if risk_analysis.get('max_losing_streak', 0) > 8:
                improvements.append("تحسين إدارة المخاطر خلال فترات الخسارة")
            
            return improvements if improvements else ["الأداء جيد بشكل عام"]
            
        except Exception as e:
            return ["تحتاج إلى مزيد من التحليل"]

    def _generate_final_recommendations(self, backtest_results: Dict) -> List[str]:
        """توليد توصيات نهائية"""
        recommendations = []
        
        try:
            overall_perf = backtest_results.get('overall_performance', {})
            betting_analysis = backtest_results.get('betting_analysis', {})
            
            accuracy = overall_perf.get('overall_accuracy', 0)
            roi = overall_perf.get('overall_roi', 0)
            
            if accuracy < 0.5:
                recommendations.append("🎯 إعطاء أولوية لتحسين دقة النماذج التنبؤية")
            
            if roi < 5:
                recommendations.append("💰 تحسين استراتيجية الرهان لزيادة العوائد")
            
            # توصيات بناءً على تحليل السوق
            market_perf = betting_analysis.get('by_market', {})
            best_market = max(market_perf.items(), 
                            key=lambda x: x[1].get('roi', -100), 
                            default=(None, {}))
            
            if best_market[0] and best_market[1].get('roi', 0) > 10:
                recommendations.append(f"🚀 التركيز على رهانات {best_market[0]} لتحقيق أعلى عوائد")
            
            return recommendations
            
        except Exception as e:
            return ["مواصلة جمع البيانات وتحليل الأداء"]

    # الوظائف الحالية من الكود الأصلي (للتوافق)
    def save_training_report(self, training_history, model_metrics, filename="training_report"):
        """حفظ تقرير التدريب"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_history': training_history,
            'model_metrics': model_metrics,
            'summary': {
                'total_training_episodes': len(training_history),
                'best_genetic_fitness': max([h.get('genetic_fitness', 0) for h in training_history]) if training_history else 0,
                'final_neural_accuracy': model_metrics.get('neural_accuracy', 0),
                'final_rl_reward': model_metrics.get('rl_reward', 0)
            }
        }
        
        # حفظ كـ JSON
        json_file = f"{self.base_path}/training/{filename}.json"
        self._save_json(report, json_file)
        
        # حفظ كـ CSV للتحليل
        if training_history:
            df = pd.DataFrame(training_history)
            csv_file = f"{self.base_path}/training/{filename}.csv"
            self._save_csv(df, csv_file)
        
        return report

    def create_training_plots(self, training_data):
        """إنشاء مخططات التدريب"""
        if not training_data:
            return
            
        df = pd.DataFrame(training_data)
        
        # مخطط تطور اللياقة الجينية
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        if 'genetic_fitness' in df.columns:
            plt.plot(df['genetic_fitness'])
            plt.title('تطور اللياقة الجينية')
            plt.xlabel('الجيل')
            plt.ylabel('اللياقة')
        
        plt.subplot(2, 2, 2)
        if 'neural_accuracy' in df.columns:
            plt.plot(df['neural_accuracy'])
            plt.title('دقة النموذج العصبي')
            plt.xlabel('الحلقة')
            plt.ylabel('الدقة')
        
        plt.subplot(2, 2, 3)
        if 'rl_reward' in df.columns:
            plt.plot(df['rl_reward'])
            plt.title('مكافأة التعلم المعزز')
            plt.xlabel('الحلقة')
            plt.ylabel('المكافأة')
        
        plt.tight_layout()
        plot_file = f"{self.base_path}/visualizations/charts/training_evolution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def save_predictions_report(self, predictions, matches_data, filename="predictions_report"):
        """حفظ تقرير التنبؤات"""
        predictions_data = []
        for match, pred in zip(matches_data, predictions):
            if 'actual_result' in match:
                prediction_record = {
                    'match': f"{match['context']['home_team']} vs {match['context']['away_team']}",
                    'predicted_home_goals': pred['home_goals'],
                    'predicted_away_goals': pred['away_goals'], 
                    'confidence': pred.get('ensemble_confidence', 0.5),
                    'actual_home_goals': match['actual_result']['home_goals'],
                    'actual_away_goals': match['actual_result']['away_goals'],
                    'timestamp': datetime.now().isoformat()
                }
                predictions_data.append(prediction_record)
        
        if predictions_data:
            csv_file = f"{self.base_path}/predictions/{filename}.csv"
            self._save_csv(pd.DataFrame(predictions_data), csv_file)
            
            # إنشاء مخطط التنبؤات vs النتائج الفعلية
            self.create_predictions_plot(pd.DataFrame(predictions_data))
        
        return predictions_data

    def create_predictions_plot(self, predictions_df):
        """مخطط مقارنة التنبؤات بالنتائج الفعلية"""
        if len(predictions_df) == 0:
            return
            
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.scatter(predictions_df['predicted_home_goals'], predictions_df['actual_home_goals'], alpha=0.6)
        plt.plot([0, 5], [0, 5], 'r--')
        plt.xlabel('الأهداف المتوقعة (منزل)')
        plt.ylabel('الأهداف الفعلية (منزل)')
        plt.title('دقة تنبؤ الأهداف - الفريق المنزل')
        
        plt.subplot(2, 2, 2)
        plt.scatter(predictions_df['predicted_away_goals'], predictions_df['actual_away_goals'], alpha=0.6)
        plt.plot([0, 5], [0, 5], 'r--')
        plt.xlabel('الأهداف المتوقعة (ضيف)')
        plt.ylabel('الأهداف الفعلية (ضيف)')
        plt.title('دقة تنبؤ الأهداف - الفريق الضيف')
        
        plt.subplot(2, 2, 3)
        accuracy_by_confidence = predictions_df.groupby(pd.cut(predictions_df['confidence'], bins=5)).apply(
            lambda x: ((x['predicted_home_goals'] == x['actual_home_goals']) & 
                      (x['predicted_away_goals'] == x['actual_away_goals'])).mean()
        )
        accuracy_by_confidence.plot(kind='bar')
        plt.title('دقة التنبؤ حسب مستوى الثقة')
        plt.xlabel('مستوى الثقة')
        plt.ylabel('نسبة الدقة')
        
        plt.tight_layout()
        plot_file = f"{self.base_path}/visualizations/charts/predictions_accuracy.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def save_betting_recommendations(self, betting_opportunities, portfolio_performance, filename="betting_recommendations"):
        """حفظ توصيات الرهان"""
        if not betting_opportunities:
            return
            
        # حفظ التوصيات
        recommendations_df = pd.DataFrame(betting_opportunities)
        csv_file = f"{self.base_path}/betting_recommendations/{filename}.csv"
        self._save_csv(recommendations_df, csv_file)
        
        # حفظ أداء المحفظة
        performance_df = pd.DataFrame([portfolio_performance])
        performance_csv = f"{self.base_path}/performance_analysis/portfolio_performance.csv"
        self._save_csv(performance_df, performance_csv)
        
        # إنشاء مخططات الرهان
        self.create_betting_visualizations(betting_opportunities, portfolio_performance)

    def create_betting_visualizations(self, betting_opportunities, portfolio_performance):
        """مخططات تحليل الرهان"""
        if not betting_opportunities:
            return
            
        df = pd.DataFrame(betting_opportunities)
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 2, 1)
        df['bet_type'].value_counts().plot(kind='bar')
        plt.title('توزيع أنواع الرهانات')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 2)
        plt.scatter(df['confidence'], df['recommended_stake'], alpha=0.6)
        plt.xlabel('مستوى الثقة')
        plt.ylabel('مبلغ الرهان المقترح')
        plt.title('العلاقة بين الثقة ومبلغ الرهان')
        
        plt.subplot(3, 2, 3)
        df['odds'].plot(kind='hist', bins=20)
        plt.title('توزيع نسب الرهان (Odds)')
        plt.xlabel('نسبة الرهان')
        
        plt.subplot(3, 2, 4)
        stake_by_match = df.groupby('match')['recommended_stake'].sum().sort_values(ascending=False)
        stake_by_match.head(10).plot(kind='bar')
        plt.title('أكبر 10 مباريات من حيث مبلغ الرهان')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_file = f"{self.base_path}/visualizations/charts/betting_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_weekly_report(self, system_dashboard, weekly_plan, betting_opportunities):
        """تقرير أسبوعي شامل"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'week_number': weekly_plan.get('week_number', 0),
            'system_health': system_dashboard.get('system_health', {}),
            'portfolio_performance': system_dashboard.get('portfolio_performance', {}),
            'weekly_plan': weekly_plan,
            'betting_opportunities_count': len(betting_opportunities),
            'total_recommended_stake': sum(opp['recommended_stake'] for opp in betting_opportunities),
            'average_confidence': np.mean([opp['confidence'] for opp in betting_opportunities]) if betting_opportunities else 0
        }
        
        report_file = f"{self.base_path}/weekly_reports/week_{weekly_plan.get('week_number', 0)}_report.json"
        self._save_json(report, report_file)
        
        return report