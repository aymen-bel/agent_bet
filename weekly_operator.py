# weekly_operator.py - مدير التشغيل الأسبوعي للموسم الحالي
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from advanced_system import EnhancedBettingSystem

class WeeklyOperationManager:
    def __init__(self, system: EnhancedBettingSystem):
        self.system = system
        self.logger = logging.getLogger(__name__)
    
    def execute_weekly_workflow(self) -> Dict:
        """تنفيذ سير العمل الأسبوعي الكامل"""
        week_report = {
            'week_number': datetime.now().isocalendar()[1],
            'start_date': datetime.now().isoformat(),
            'daily_operations': {},
            'weekly_summary': {}
        }
        
        try:
            # الاثنين: التحليل والإعداد
            monday_result = self._monday_operations()
            week_report['daily_operations']['monday'] = monday_result
            
            # الثلاثاء: التخطيط
            tuesday_result = self._tuesday_operations()
            week_report['daily_operations']['tuesday'] = tuesday_result
            
            # الأربعاء: المراجعة
            wednesday_result = self._wednesday_operations()
            week_report['daily_operations']['wednesday'] = wednesday_result
            
            # الخميس: التنفيذ المبكر
            thursday_result = self._thursday_operations()
            week_report['daily_operations']['thursday'] = thursday_result
            
            # الجمعة: التنفيذ النهائي
            friday_result = self._friday_operations()
            week_report['daily_operations']['friday'] = friday_result
            
            # عطلة نهاية الأسبوع: المتابعة
            weekend_result = self._weekend_operations()
            week_report['daily_operations']['weekend'] = weekend_result
            
            # تقرير أسبوعي
            week_report['weekly_summary'] = self._generate_weekly_summary()
            
            self.logger.info("✅ اكتمل التنفيذ الأسبوعي بنجاح")
            return week_report
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التنفيذ الأسبوعي: {e}")
            week_report['error'] = str(e)
            return week_report
    
    def _monday_operations(self) -> Dict:
        """عمليات يوم الاثنين"""
        self.logger.info("📊 بدء عمليات الاثنين...")
        
        operations = {
            'data_loading': 'تحميل بيانات نهاية الأسبوع',
            'performance_review': 'مراجعة أداء الرهانات السابقة',
            'system_calibration': 'إعادة معايرة النماذج إذا لزم الأمر'
        }
        
        # تحميل البيانات الجديدة
        try:
            self.system.data_enricher.load_and_combine_data()
            operations['data_loading_status'] = 'SUCCESS'
        except Exception as e:
            operations['data_loading_status'] = f'FAILED: {e}'
        
        # مراجعة الأداء
        portfolio_perf = self.system.portfolio_manager.get_portfolio_performance()
        operations['performance_review'] = portfolio_perf
        
        # معايرة النظام إذا كان الأداء ضعيفاً
        if portfolio_perf.get('win_rate', 0) < 0.5:
            operations['calibration_needed'] = True
            # إعادة تدريب سريع
            self.system.train_for_current_season(training_episodes=100, genetic_generations=20)
        else:
            operations['calibration_needed'] = False
        
        return operations
    
    def _tuesday_operations(self) -> Dict:
        """عمليات يوم الثلاثاء"""
        self.logger.info("📈 بدء عمليات الثلاثاء...")
        
        operations = {
            'match_analysis': 'تحليل المباريات القادمة',
            'initial_predictions': 'توليد التنبؤات الأولية',
            'risk_assessment': 'تقييم المخاطر'
        }
        
        # تحليل المباريات القادمة
        try:
            upcoming_matches = self._get_upcoming_matches()
            operations['matches_count'] = len(upcoming_matches)
            
            # توليد التنبؤات الأولية
            initial_predictions = []
            for match in upcoming_matches[:10]:  # تحليل أول 10 مباريات
                prediction = self.system.ensemble_prediction(
                    match['team_metrics'],
                    match['opponent_metrics'],
                    match['context']
                )
                initial_predictions.append({
                    'match': f"{match['context'].get('home_team')} vs {match['context'].get('away_team')}",
                    'prediction': f"{prediction['home_goals']}-{prediction['away_goals']}",
                    'confidence': prediction['ensemble_confidence']
                })
            
            operations['initial_predictions'] = initial_predictions
            operations['analysis_status'] = 'SUCCESS'
            
        except Exception as e:
            operations['analysis_status'] = f'FAILED: {e}'
        
        # تقييم المخاطر
        risk_status = self.system.portfolio_manager.risk_management_check()
        operations['risk_assessment'] = risk_status
        
        return operations
    
    def _wednesday_operations(self) -> Dict:
        """عمليات يوم الأربعاء"""
        self.logger.info("🔍 بدء عمليات الأربعاء...")
        
        operations = {
            'injury_updates': 'مراجعة تحديثات الإصابات',
            'final_predictions': 'تحديث التنبؤات النهائية',
            'betting_opportunities': 'تحديد فرص الرهان'
        }
        
        try:
            # الحصول على فرص الرهان النهائية
            upcoming_matches = self._get_upcoming_matches()
            betting_ops = self.system.generate_intelligent_bets(upcoming_matches)
            
            operations['betting_opportunities'] = betting_ops
            operations['opportunities_count'] = betting_ops.get('opportunities_count', 0)
            operations['total_recommended_stake'] = betting_ops.get('total_recommended_stake', 0)
            
            operations['status'] = 'SUCCESS'
            
        except Exception as e:
            operations['status'] = f'FAILED: {e}'
        
        return operations
    
    def _thursday_operations(self) -> Dict:
        """عمليات يوم الخميس"""
        self.logger.info("💸 بدء عمليات الخميس...")
        
        operations = {
            'early_betting': 'تنفيذ الرهانات المبكرة',
            'odds_monitoring': 'مراقبة تغيرات odds'
        }
        
        try:
            # تنفيذ الرهانات المبكرة
            betting_ops = self.system.portfolio_manager.bet_history[-1] if self.system.portfolio_manager.bet_history else {}
            
            if betting_ops.get('betting_opportunities'):
                early_bets = [bet for bet in betting_ops['betting_opportunities'] 
                             if bet.get('odds', 0) > 2.0]  # رهانات ذات odds جيدة
                
                executed = self.system.execute_betting_strategy(early_bets[:3])  # أول 3 رهانات
                operations['early_bets_executed'] = executed
                operations['early_bets_count'] = executed.get('bets_count', 0)
            else:
                operations['early_bets_executed'] = 'NO_OPPORTUNITIES'
            
            operations['status'] = 'SUCCESS'
            
        except Exception as e:
            operations['status'] = f'FAILED: {e}'
        
        return operations
    
    def _friday_operations(self) -> Dict:
        """عمليات يوم الجمعة"""
        self.logger.info("🎯 بدء عمليات الجمعة...")
        
        operations = {
            'final_betting': 'تنفيذ الرهانات المتبقية',
            'risk_final_check': 'مراجعة نهائية للمخاطر'
        }
        
        try:
            # تنفيذ الرهانات المتبقية
            betting_ops = self.system.portfolio_manager.bet_history[-1] if self.system.portfolio_manager.bet_history else {}
            
            if betting_ops.get('betting_opportunities'):
                remaining_bets = betting_ops['betting_opportunities'][3:]  # الرهانات المتبقية
                executed = self.system.execute_betting_strategy(remaining_bets)
                operations['final_bets_executed'] = executed
                operations['final_bets_count'] = executed.get('bets_count', 0)
            else:
                operations['final_bets_executed'] = 'NO_OPPORTUNITIES'
            
            # مراجعة المخاطر النهائية
            risk_status = self.system.portfolio_manager.risk_management_check()
            operations['final_risk_check'] = risk_status
            
            operations['status'] = 'SUCCESS'
            
        except Exception as e:
            operations['status'] = f'FAILED: {e}'
        
        return operations
    
    def _weekend_operations(self) -> Dict:
        """عمليات عطلة نهاية الأسبوع"""
        self.logger.info("📋 بدء عمليات نهاية الأسبوع...")
        
        operations = {
            'match_monitoring': 'متابعة المباريات والنتائج',
            'results_recording': 'تسجيل نتائج الرهانات',
            'portfolio_update': 'تحديث المحفظة',
            'weekly_report': 'إعداد التقرير الأسبوعي'
        }
        
        try:
            # محاكاة تحديث النتائج
            pending_bets = [bet for bet in self.system.portfolio_manager.bet_history 
                          if bet.get('status') == 'PENDING']
            
            for bet in pending_bets:
                # محاكاة النتائج (في الواقع الفعلي، تأتي من مصدر حقيقي)
                bet['status'] = 'SETTLED'
                bet['is_win'] = np.random.random() > 0.4  # 60% فرصة فوز
                bet['profit'] = bet['stake'] * (bet['odds'] - 1) if bet['is_win'] else -bet['stake']
                
                # تحديث المحفظة
                self.system.portfolio_manager.update_portfolio(bet)
            
            operations['bets_settled'] = len(pending_bets)
            operations['portfolio_updated'] = True
            
            operations['status'] = 'SUCCESS'
            
        except Exception as e:
            operations['status'] = f'FAILED: {e}'
        
        return operations
    
    def _generate_weekly_summary(self) -> Dict:
        """توليد تقرير أسبوعي"""
        portfolio_perf = self.system.portfolio_manager.get_portfolio_performance()
        dashboard = self.system.get_current_season_dashboard()
        
        return {
            'weekly_performance': {
                'bets_placed': portfolio_perf['total_bets'],
                'win_rate': portfolio_perf['win_rate'],
                'weekly_profit': portfolio_perf['total_profit'],
                'weekly_roi': portfolio_perf['roi']
            },
            'portfolio_status': {
                'current_capital': portfolio_perf['current_capital'],
                'overall_growth': portfolio_perf['overall_growth'],
                'risk_level': dashboard['risk_management'].get('warnings', [])
            },
            'system_health': dashboard['system_health'],
            'recommendations': self._generate_weekly_recommendations(portfolio_perf)
        }
    
    def _generate_weekly_recommendations(self, performance: Dict) -> List[str]:
        """توليد توصيات أسبوعية"""
        recommendations = []
        
        if performance['win_rate'] < 0.5:
            recommendations.append("🔄 فكر في إعادة تدريب النماذج لتحسين الدقة")
        
        if performance['roi'] < 0:
            recommendations.append("💰 خفض حجم الرهانات حتى تحسن الأداء")
        
        if len(performance.get('risk_warnings', [])) > 0:
            recommendations.append("⚠️  راجع استراتيجية إدارة المخاطر")
        
        if performance['total_bets'] < 5:
            recommendations.append("🎯 زد من فرص الرهان ولكن بحكمة")
        
        return recommendations
    
    def _get_upcoming_matches(self) -> List[Dict]:
        """الحصول على المباريات القادمة (محاكاة)"""
        # في التنفيذ الفعلي، سيتم جلب هذه البيانات من مصدر حقيقي
        sample_matches = []
        
        for i in range(15):
            sample_matches.append({
                'team_metrics': {
                    'points_per_match': np.random.uniform(1.0, 2.5),
                    'win_rate': np.random.uniform(0.3, 0.8),
                    'goal_difference': np.random.randint(-10, 30),
                    'goals_per_match': np.random.uniform(0.8, 2.5),
                    'conceded_per_match': np.random.uniform(0.5, 2.0),
                    'current_form': np.random.uniform(0.2, 0.9),
                    'defensive_efficiency': np.random.uniform(0.4, 0.9)
                },
                'opponent_metrics': {
                    'points_per_match': np.random.uniform(1.0, 2.5),
                    'win_rate': np.random.uniform(0.3, 0.8),
                    'goal_difference': np.random.randint(-10, 30),
                    'goals_per_match': np.random.uniform(0.8, 2.5),
                    'conceded_per_match': np.random.uniform(0.5, 2.0),
                    'current_form': np.random.uniform(0.2, 0.9),
                    'defensive_efficiency': np.random.uniform(0.4, 0.9)
                },
                'context': {
                    'home_team': f'Team_{i}A',
                    'away_team': f'Team_{i}B',
                    'home_advantage': np.random.uniform(1.0, 1.3),
                    'match_importance': np.random.uniform(0.8, 1.2)
                },
                'actual_result': {
                    'home_goals': np.random.randint(0, 4),
                    'away_goals': np.random.randint(0, 4)
                }
            })
        
        return sample_matches

def run_weekly_operation():
    """تشغيل المدير الأسبوعي"""
    system = EnhancedBettingSystem("data/football-data")
    operator = WeeklyOperationManager(system)
    
    print("📅 بدء التشغيل الأسبوعي للموسم الحالي...")
    print("=" * 50)
    
    # تدريب سريع إذا needed
    system.train_for_current_season(training_episodes=150, genetic_generations=25)
    
    # تنفيذ سير العمل الأسبوعي
    weekly_report = operator.execute_weekly_workflow()
    
    # عرض النتائج
    print(f"\n📊 التقرير الأسبوعي #{weekly_report['week_number']}")
    print(f"📈 الأداء الأسبوعي:")
    summary = weekly_report['weekly_summary']
    perf = summary['weekly_performance']
    
    print(f"• الرهانات المنفذة: {perf['bets_placed']}")
    print(f"• معدل الفوز: {perf['win_rate']:.1%}")
    print(f"• الربح الأسبوعي: {perf['weekly_profit']:.2f}$")
    print(f"• عائد الاستثمار: {perf['weekly_roi']:.1f}%")
    
    print(f"\n💡 التوصيات:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    # حفظ التقرير
    with open(f'weekly_reports/week_{weekly_report["week_number"]}.json', 'w', encoding='utf-8') as f:
        json.dump(weekly_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 تم حفظ التقرير الأسبوعي")

if __name__ == "__main__":
    run_weekly_operation()