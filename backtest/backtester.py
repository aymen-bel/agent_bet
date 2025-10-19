# backtester.py
"""
نظام Backtest المتكامل لتقييم أداء النماذج والرهان
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum

class BacktestMode(Enum):
    HISTORICAL = "historical"
    CURRENT_SEASON = "current_season"
    FUTURE = "future"
    COMPREHENSIVE = "comprehensive"

class BettingResultType(Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    HALF_WIN = "half_win"
    HALF_LOSS = "half_loss"

class AdvancedBacktester:
    """نظام Backtest متقدم لتقييم أداء النماذج والرهان"""
    
    def __init__(self, main_system, output_manager):
        self.main_system = main_system
        self.output_manager = output_manager
        self.backtest_results = {}
        self.betting_performance = {}
        self.model_accuracy = {}
        
        # إعداد التسجيل
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_backtest(self, start_season: str = "2020-2021", 
                                 end_season: str = "2025-2026",
                                 initial_bankroll: float = 10000.0) -> Dict:
        """تشغيل backtest شامل لجميع المواسم"""
        
        self.logger.info("🔄 بدء Backtest الشامل للمواسم المتعددة...")
        
        comprehensive_results = {
            'backtest_id': f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'start_season': start_season,
                'end_season': end_season,
                'initial_bankroll': initial_bankroll,
                'betting_strategy': 'comprehensive'
            },
            'seasonal_results': {},
            'overall_performance': {},
            'betting_analysis': {},
            'model_performance': {}
        }
        
        try:
            # 1. Backtest للمواسم التاريخية
            historical_results = self._run_historical_backtest(
                start_season, "2023-2024", initial_bankroll
            )
            comprehensive_results['seasonal_results']['historical'] = historical_results
            
            # 2. Backtest للموسم الحالي
            current_results = self._run_current_season_backtest(
                "2024-2025", initial_bankroll
            )
            comprehensive_results['seasonal_results']['current'] = current_results
            
            # 3. Backtest للموسم المستقبلي
            future_results = self._run_future_backtest(
                "2025-2026", initial_bankroll
            )
            comprehensive_results['seasonal_results']['future'] = future_results
            
            # 4. تحليل شامل للأداء
            comprehensive_results['overall_performance'] = self._calculate_overall_performance(
                comprehensive_results['seasonal_results']
            )
            
            # 5. تحليل أداء الرهان
            comprehensive_results['betting_analysis'] = self._analyze_betting_performance(
                comprehensive_results['seasonal_results']
            )
            
            # 6. تقييم أداء النماذج
            comprehensive_results['model_performance'] = self._evaluate_model_performance(
                comprehensive_results['seasonal_results']
            )
            
            comprehensive_results['end_time'] = datetime.now().isoformat()
            comprehensive_results['success'] = True
            
            # حفظ النتائج
            self._save_comprehensive_backtest_results(comprehensive_results)
            
            self.logger.info("✅ اكتمل Backtest الشامل")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"❌ فشل في Backtest الشامل: {e}")
            comprehensive_results['error'] = str(e)
            comprehensive_results['success'] = False
            return comprehensive_results
    
    def _run_historical_backtest(self, start_season: str, end_season: str, 
                               initial_bankroll: float) -> Dict:
        """تشغيل Backtest للمواسم التاريخية"""
        
        self.logger.info(f"📊 بدء Backtest للمواسم {start_season} إلى {end_season}")
        
        results = {
            'period': f"{start_season}_{end_season}",
            'total_matches': 0,
            'model_predictions': {},
            'betting_results': {},
            'bankroll_evolution': [],
            'performance_metrics': {}
        }
        
        # تحميل البيانات التاريخية
        historical_data = self._load_historical_data_for_backtest(start_season, end_season)
        
        if not historical_data:
            self.logger.warning("⚠️ لا توجد بيانات تاريخية للـ Backtest")
            return results
        
        results['total_matches'] = len(historical_data)
        
        # محاكاة التنبؤ والرهان لكل مباراة
        current_bankroll = initial_bankroll
        betting_results = []
        model_predictions = []
        
        for match_data in historical_data[:500]:  # عينة للاختبار
            try:
                # توليد التنبؤ (بدون النظر للنتيجة الفعلية)
                prediction = self._generate_prediction_for_backtest(match_data)
                
                if prediction:
                    model_predictions.append(prediction)
                    
                    # محاكاة الرهان
                    bet_results = self._simulate_betting_for_match(
                        prediction, match_data, current_bankroll
                    )
                    
                    if bet_results:
                        # تحديث الرصيد
                        current_bankroll += bet_results['net_profit']
                        
                        betting_record = {
                            'match_info': {
                                'home_team': match_data.get('HomeTeam'),
                                'away_team': match_data.get('AwayTeam'),
                                'date': match_data.get('Date'),
                                'season': match_data.get('Season')
                            },
                            'prediction': prediction,
                            'actual_result': {
                                'home_goals': match_data.get('FTHG'),
                                'away_goals': match_data.get('FTAG')
                            },
                            'betting_decision': bet_results['betting_decision'],
                            'stake': bet_results['stake'],
                            'odds': bet_results['odds'],
                            'result': bet_results['result'],
                            'profit': bet_results['profit'],
                            'net_profit': bet_results['net_profit'],
                            'bankroll_after': current_bankroll
                        }
                        
                        betting_results.append(betting_record)
                        
                        # حفظ تطور الرصيد
                        if len(betting_results) % 10 == 0:
                            results['bankroll_evolution'].append({
                                'match_count': len(betting_results),
                                'bankroll': current_bankroll,
                                'timestamp': datetime.now().isoformat()
                            })
                
            except Exception as e:
                self.logger.warning(f"⚠️ خطأ في معالجة المباراة: {e}")
                continue
        
        # حساب مقاييس الأداء
        results['model_predictions'] = model_predictions
        results['betting_results'] = betting_results
        results['performance_metrics'] = self._calculate_performance_metrics(
            model_predictions, betting_results, initial_bankroll, current_bankroll
        )
        
        return results
    
    def _run_current_season_backtest(self, season: str, initial_bankroll: float) -> Dict:
        """تشغيل Backtest للموسم الحالي"""
        
        self.logger.info(f"🎯 بدء Backtest للموسم الحالي {season}")
        
        # تنفيذ مشابه للتاريخي مع تعديلات للموسم الحالي
        return self._run_historical_backtest(season, season, initial_bankroll)
    
    def _run_future_backtest(self, season: str, initial_bankroll: float) -> Dict:
        """تشغيل Backtest للموسم المستقبلي"""
        
        self.logger.info(f"🔮 بدء Backtest للموسم المستقبلي {season}")
        
        results = {
            'period': season,
            'total_matches': 0,
            'model_predictions': {},
            'betting_results': {},
            'bankroll_evolution': [],
            'performance_metrics': {}
        }
        
        # استخدام نظام التنبؤ الرئيسي للتنبؤات المستقبلية
        try:
            future_predictions = self.main_system.run_comprehensive_prediction_pipeline(weeks_ahead=38)
            
            if future_predictions and 'predictions' in future_predictions:
                predictions = future_predictions['predictions']
                results['total_matches'] = len(predictions)
                
                current_bankroll = initial_bankroll
                betting_results = []
                
                for prediction_data in predictions:
                    try:
                        # محاكاة الرهان على التنبؤات المستقبلية
                        bet_results = self._simulate_future_betting(
                            prediction_data, current_bankroll
                        )
                        
                        if bet_results:
                            current_bankroll += bet_results['net_profit']
                            
                            betting_record = {
                                'match_info': prediction_data.get('match_info', {}),
                                'prediction': prediction_data.get('comprehensive_prediction', {}),
                                'betting_decision': bet_results['betting_decision'],
                                'stake': bet_results['stake'],
                                'odds': bet_results['odds'],
                                'expected_profit': bet_results['expected_profit'],
                                'bankroll_after': current_bankroll
                            }
                            
                            betting_results.append(betting_record)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ خطأ في الرهان المستقبلي: {e}")
                        continue
                
                results['betting_results'] = betting_results
                results['performance_metrics'] = self._calculate_future_performance_metrics(
                    betting_results, initial_bankroll, current_bankroll
                )
        
        except Exception as e:
            self.logger.error(f"❌ خطأ في Backtest المستقبلي: {e}")
        
        return results
    
    def _load_historical_data_for_backtest(self, start_season: str, end_season: str) -> List[Dict]:
        """تحميل البيانات التاريخية للـ Backtest"""
        
        historical_data = []
        
        try:
            # تحميل البيانات من ملفات المواسم
            base_path = "data/seasons"
            seasons = self._generate_season_range(start_season, end_season)
            
            for season in seasons:
                filename = f"england_E0_{season[:4]}.csv"
                filepath = os.path.join(base_path, filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    
                    # إضافة عمود الموسم
                    df['Season'] = season
                    
                    # تحويل إلى قائمة dictionaries
                    season_data = df.to_dict('records')
                    historical_data.extend(season_data)
                    
                    self.logger.info(f"✅ تم تحميل {len(season_data)} مباراة من {season}")
                else:
                    self.logger.warning(f"⚠️ ملف غير موجود: {filepath}")
        
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل البيانات التاريخية: {e}")
        
        return historical_data
    
    def _generate_season_range(self, start_season: str, end_season: str) -> List[str]:
        """توليد قائمة المواسم بين بداية ونهاية"""
        
        seasons = []
        start_year = int(start_season[:4])
        end_year = int(end_season[:4])
        
        for year in range(start_year, end_year + 1):
            seasons.append(f"{year}-{year+1}")
        
        return seasons
    
    def _generate_prediction_for_backtest(self, match_data: Dict) -> Optional[Dict]:
        """توليد تنبؤ للمباراة (بدون النظر للنتيجة الفعلية)"""
        
        try:
            home_team = match_data.get('HomeTeam')
            away_team = match_data.get('AwayTeam')
            
            if not home_team or not away_team:
                return None
            
            # استخدام نظام التنبؤ الرئيسي
            if hasattr(self.main_system, 'prediction_engine') and self.main_system.prediction_engine:
                prediction = self.main_system.prediction_engine.generate_comprehensive_prediction(
                    home_team, away_team, "home"
                )
                return prediction
            else:
                # تنبؤ أساسي إذا لم يكن النظام جاهزاً
                return self._generate_basic_prediction(match_data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في توليد التنبؤ: {e}")
            return None
    
    def _generate_basic_prediction(self, match_data: Dict) -> Dict:
        """توليد تنبؤ أساسي للمباراة"""
        
        home_goals = match_data.get('FTHG', 0)
        away_goals = match_data.get('FTAG', 0)
        
        # استخدام البيانات التاريخية للتنبؤ الأساسي
        return {
            'home_team': match_data.get('HomeTeam'),
            'away_team': match_data.get('AwayTeam'),
            'consensus_prediction': {
                'home_goals': max(0, int(np.random.poisson(1.5))),
                'away_goals': max(0, int(np.random.poisson(1.2))),
                'confidence': np.random.uniform(0.3, 0.8),
                'consensus_type': 'basic_backtest'
            },
            'betting_predictions': {
                '1x2_market': {
                    'home_win': {'probability': 0.4, 'odds': 2.1},
                    'draw': {'probability': 0.3, 'odds': 3.2},
                    'away_win': {'probability': 0.3, 'odds': 3.5}
                }
            },
            'recommendations': ['تنبؤ أساسي للـ Backtest']
        }
    
    def _simulate_betting_for_match(self, prediction: Dict, match_data: Dict, 
                                  current_bankroll: float) -> Optional[Dict]:
        """محاكاة الرهان على مباراة معينة"""
        
        try:
            # استراتيجية الرهان (1% من الرصيد)
            stake = current_bankroll * 0.01
            
            # تحديد قرار الرهان بناءً على التنبؤ
            betting_decision = self._determine_betting_decision(prediction)
            
            if not betting_decision:
                return None
            
            # الحصول on odds حقيقية أو محاكاة
            odds = self._get_simulated_odds(prediction, betting_decision)
            
            # محاكاة النتيجة
            actual_home_goals = match_data.get('FTHG', 0)
            actual_away_goals = match_data.get('FTAG', 0)
            
            # تحديد نتيجة الرهان
            bet_result = self._determine_bet_result(
                betting_decision, actual_home_goals, actual_away_goals, odds
            )
            
            return {
                'betting_decision': betting_decision,
                'stake': stake,
                'odds': odds,
                'result': bet_result['type'],
                'profit': bet_result['profit'],
                'net_profit': bet_result['net_profit']
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في محاكاة الرهان: {e}")
            return None
    
    def _determine_betting_decision(self, prediction: Dict) -> Optional[Dict]:
        """تحديد قرار الرهان الأمثل بناءً على التنبؤ"""
        
        try:
            betting_predictions = prediction.get('betting_predictions', {})
            
            if not betting_predictions:
                return None
            
            # البحث عن أفضل رهان بناءً على القيمة المتوقعة
            best_bet = None
            best_expected_value = 0
            
            # تحليل رهانات 1X2
            if '1x2_market' in betting_predictions:
                market = betting_predictions['1x2_market']
                
                for outcome, data in market.items():
                    probability = data.get('probability', 0)
                    odds = data.get('odds', 1.0)
                    
                    expected_value = (probability * (odds - 1)) - (1 - probability)
                    
                    if expected_value > best_expected_value and expected_value > 0.05:
                        best_expected_value = expected_value
                        best_bet = {
                            'market': '1x2',
                            'selection': outcome,
                            'probability': probability,
                            'odds': odds,
                            'expected_value': expected_value
                        }
            
            # تحليل رهانات BTTS
            if 'btts' in betting_predictions and not best_bet:
                btts_data = betting_predictions['btts']
                yes_prob = btts_data.get('yes', {}).get('probability', 0)
                yes_odds = btts_data.get('yes', {}).get('odds', 1.8)
                
                ev_yes = (yes_prob * (yes_odds - 1)) - (1 - yes_prob)
                
                if ev_yes > best_expected_value and ev_yes > 0.05:
                    best_expected_value = ev_yes
                    best_bet = {
                        'market': 'btts',
                        'selection': 'yes',
                        'probability': yes_prob,
                        'odds': yes_odds,
                        'expected_value': ev_yes
                    }
            
            return best_bet
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحديد قرار الرهان: {e}")
            return None
    
    def _get_simulated_odds(self, prediction: Dict, betting_decision: Dict) -> float:
        """الحصول on odds محاكاة واقعية"""
        
        try:
            # استخدام odds من التنبؤ إذا كانت متاحة
            if 'odds' in betting_decision:
                return betting_decision['odds']
            
            # توليد odds واقعية بناءً على الاحتمال
            probability = betting_decision.get('probability', 0.5)
            margin = 0.05  # هامش الربح للكتاب
            
            # تجنب القسمة على الصفر
            if probability <= 0:
                probability = 0.01
            elif probability >= 1:
                probability = 0.99
                
            fair_odds = 1.0 / probability
            simulated_odds = fair_odds * (1 - margin)
            
            return max(1.1, min(10.0, simulated_odds))
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في محاكاة odds: {e}")
            return 2.0  # قيمة افتراضية
    
    def _determine_bet_result(self, betting_decision: Dict, actual_home: int, 
                            actual_away: int, odds: float) -> Dict:
        """تحديد نتيجة الرهان بناءً على النتيجة الفعلية"""
        
        try:
            market = betting_decision['market']
            selection = betting_decision['selection']
            stake = 100  # قيمة افتراضية للستايك
            
            is_win = False
            
            if market == '1x2':
                if selection == 'home_win':
                    is_win = actual_home > actual_away
                elif selection == 'draw':
                    is_win = actual_home == actual_away
                elif selection == 'away_win':
                    is_win = actual_away > actual_home
            
            elif market == 'btts':
                if selection == 'yes':
                    is_win = actual_home > 0 and actual_away > 0
                else:  # no
                    is_win = actual_home == 0 or actual_away == 0
            
            elif market == 'over_under':
                line = float(selection.split('_')[1])
                total_goals = actual_home + actual_away
                
                if 'over' in selection:
                    is_win = total_goals > line
                else:  # under
                    is_win = total_goals < line
            
            if is_win:
                profit = stake * (odds - 1)
                net_profit = profit
                result_type = BettingResultType.WIN
            else:
                profit = -stake
                net_profit = -stake
                result_type = BettingResultType.LOSS
            
            return {
                'type': result_type,
                'profit': profit,
                'net_profit': net_profit,
                'stake': stake
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحديد نتيجة الرهان: {e}")
            return {
                'type': BettingResultType.LOSS,
                'profit': -100,
                'net_profit': -100,
                'stake': 100
            }
    
    def _calculate_performance_metrics(self, predictions: List, betting_results: List,
                                    initial_bankroll: float, final_bankroll: float) -> Dict:
        """حساب مقاييس أداء الـ Backtest"""
        
        try:
            # مقاييس دقة النموذج
            correct_score_predictions = 0
            correct_result_predictions = 0
            total_predictions = len(predictions)
            
            for pred in predictions:
                if 'consensus_prediction' in pred:
                    consensus = pred['consensus_prediction']
                    pred_home = consensus.get('home_goals', 0)
                    pred_away = consensus.get('away_goals', 0)
                    
                    # البحث عن النتيجة الفعلية
                    actual_home = None
                    actual_away = None
                    
                    for bet in betting_results:
                        if (bet['match_info']['home_team'] == pred['home_team'] and
                            bet['match_info']['away_team'] == pred['away_team']):
                            actual_home = bet['actual_result']['home_goals']
                            actual_away = bet['actual_result']['away_goals']
                            break
                    
                    if actual_home is not None and actual_away is not None:
                        # دقة النتيجة
                        if pred_home == actual_home and pred_away == actual_away:
                            correct_score_predictions += 1
                        
                        # دقة النتيجة (فوز/تعادل/خسارة)
                        pred_result = 'H' if pred_home > pred_away else 'A' if pred_away > pred_home else 'D'
                        actual_result = 'H' if actual_home > actual_away else 'A' if actual_away > actual_home else 'D'
                        
                        if pred_result == actual_result:
                            correct_result_predictions += 1
            
            # مقاييس الرهان
            total_bets = len(betting_results)
            winning_bets = len([b for b in betting_results if b['profit'] > 0])
            losing_bets = len([b for b in betting_results if b['profit'] < 0])
            push_bets = len([b for b in betting_results if b['profit'] == 0])
            
            total_profit = sum(b['profit'] for b in betting_results)
            total_stake = sum(b['stake'] for b in betting_results)
            
            if total_bets > 0:
                win_rate = winning_bets / total_bets
                roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
            else:
                win_rate = 0
                roi = 0
            
            bankroll_growth = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
            
            return {
                'model_accuracy': {
                    'score_accuracy': correct_score_predictions / total_predictions if total_predictions > 0 else 0,
                    'result_accuracy': correct_result_predictions / total_predictions if total_predictions > 0 else 0,
                    'total_predictions': total_predictions
                },
                'betting_performance': {
                    'total_bets': total_bets,
                    'winning_bets': winning_bets,
                    'losing_bets': losing_bets,
                    'push_bets': push_bets,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'total_stake': total_stake,
                    'roi': roi,
                    'bankroll_growth': bankroll_growth,
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': final_bankroll
                },
                'risk_metrics': {
                    'max_drawdown': self._calculate_max_drawdown(betting_results),
                    'sharpe_ratio': self._calculate_sharpe_ratio(betting_results),
                    'profit_factor': self._calculate_profit_factor(betting_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حساب مقاييس الأداء: {e}")
            return {}
    
    def _calculate_max_drawdown(self, betting_results: List) -> float:
        """حساب أقصى انخفاض في الرصيد"""
        
        try:
            if not betting_results:
                return 0.0
            
            bankroll_evolution = [10000]  # بداية افتراضية
            current_bankroll = 10000
            
            for bet in betting_results:
                current_bankroll += bet['net_profit']
                bankroll_evolution.append(current_bankroll)
            
            peak = bankroll_evolution[0]
            max_drawdown = 0
            
            for value in bankroll_evolution:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown * 100  # نسبة مئوية
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في حساب أقصى انخفاض: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, betting_results: List) -> float:
        """حساب نسبة شارب"""
        
        try:
            if len(betting_results) < 2:
                return 0.0
            
            returns = [bet['net_profit'] / 100 for bet in betting_results]  # عوائد نسبية
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            sharpe_ratio = avg_return / std_return
            return sharpe_ratio
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في حساب نسبة شارب: {e}")
            return 0.0
    
    def _calculate_profit_factor(self, betting_results: List) -> float:
        """حساب عامل الربحية"""
        
        try:
            total_profits = sum(b['profit'] for b in betting_results if b['profit'] > 0)
            total_losses = abs(sum(b['profit'] for b in betting_results if b['profit'] < 0))
            
            if total_losses == 0:
                return float('inf')
            
            return total_profits / total_losses
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في حساب عامل الربحية: {e}")
            return 0.0
    
    def _calculate_overall_performance(self, seasonal_results: Dict) -> Dict:
        """حساب الأداء الشامل عبر جميع المواسم"""
        
        try:
            # جمع الإحصائيات من جميع المواسم
            all_betting_results = []
            total_predictions = 0
            correct_predictions = 0
            
            for season_type, results in seasonal_results.items():
                if 'betting_results' in results:
                    all_betting_results.extend(results['betting_results'])
                
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    if 'model_accuracy' in metrics:
                        model_acc = metrics['model_accuracy']
                        total_predictions += model_acc.get('total_predictions', 0)
                        correct_predictions += int(model_acc.get('result_accuracy', 0) * model_acc.get('total_predictions', 0))
            
            # حساب الأداء الشامل
            overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # أداء الرهان الشامل
            if all_betting_results:
                total_profit = sum(b.get('profit', 0) for b in all_betting_results)
                total_stake = sum(b.get('stake', 100) for b in all_betting_results)
                winning_bets = len([b for b in all_betting_results if b.get('profit', 0) > 0])
                total_bets = len(all_betting_results)
                
                overall_roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
                overall_win_rate = winning_bets / total_bets if total_bets > 0 else 0
            else:
                overall_roi = 0
                overall_win_rate = 0
            
            return {
                'overall_accuracy': overall_accuracy,
                'overall_roi': overall_roi,
                'overall_win_rate': overall_win_rate,
                'total_bets_placed': len(all_betting_results),
                'total_predictions_made': total_predictions,
                'performance_rating': self._calculate_performance_rating(overall_accuracy, overall_roi)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حساب الأداء الشامل: {e}")
            return {}
    
    def _calculate_performance_rating(self, accuracy: float, roi: float) -> str:
        """تقييم الأداء الشامل"""
        
        if accuracy > 0.6 and roi > 15:
            return "ممتاز"
        elif accuracy > 0.55 and roi > 10:
            return "جيد جداً"
        elif accuracy > 0.5 and roi > 5:
            return "جيد"
        elif accuracy > 0.45 and roi > 0:
            return "مقبول"
        else:
            return "بحاجة للتحسين"
    
    def _analyze_betting_performance(self, seasonal_results: Dict) -> Dict:
        """تحليل أداء الرهان المفصل"""
        
        analysis = {
            'by_market': {},
            'by_season': {},
            'risk_analysis': {},
            'recommendations': []
        }
        
        try:
            # تحليل حسب نوع الرهان
            markets = ['1x2', 'btts', 'over_under', 'double_chance']
            
            for market in markets:
                market_results = []
                
                for season_type, results in seasonal_results.items():
                    if 'betting_results' in results:
                        market_bets = [b for b in results['betting_results'] 
                                    if b.get('betting_decision', {}).get('market') == market]
                        market_results.extend(market_bets)
                
                if market_results:
                    analysis['by_market'][market] = self._analyze_market_performance(market_results)
            
            # تحليل حسب الموسم
            for season_type, results in seasonal_results.items():
                if 'performance_metrics' in results:
                    analysis['by_season'][season_type] = results['performance_metrics']
            
            # تحليل المخاطر
            analysis['risk_analysis'] = self._analyze_risk_factors(seasonal_results)
            
            # توليد التوصيات
            analysis['recommendations'] = self._generate_betting_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحليل أداء الرهان: {e}")
            return analysis
    
    def _analyze_market_performance(self, market_results: List) -> Dict:
        """تحليل أداء نوع رهان محدد"""
        
        try:
            total_bets = len(market_results)
            winning_bets = len([b for b in market_results if b.get('profit', 0) > 0])
            total_profit = sum(b.get('profit', 0) for b in market_results)
            total_stake = sum(b.get('stake', 100) for b in market_results)
            
            win_rate = winning_bets / total_bets if total_bets > 0 else 0
            roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
            
            return {
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'roi': roi,
                'average_odds': np.mean([b.get('odds', 2.0) for b in market_results]),
                'performance_rating': 'ممتاز' if roi > 10 else 'جيد' if roi > 5 else 'ضعيف'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحليل أداء السوق: {e}")
            return {}
    
    def _analyze_risk_factors(self, seasonal_results: Dict) -> Dict:
        """تحليل عوامل المخاطرة"""
        
        risk_analysis = {
            'volatility': 0,
            'consistency': 0,
            'drawdown_analysis': {},
            'betting_patterns': {}
        }
        
        try:
            # جمع جميع نتائج الرهان
            all_results = []
            for season_type, results in seasonal_results.items():
                if 'betting_results' in results:
                    all_results.extend(results['betting_results'])
            
            if not all_results:
                return risk_analysis
            
            # حساب التقلبية
            profits = [b.get('profit', 0) for b in all_results]
            risk_analysis['volatility'] = np.std(profits)
            
            # حساب الاتساق
            winning_streak = 0
            losing_streak = 0
            max_winning_streak = 0
            max_losing_streak = 0
            
            for result in all_results:
                profit = result.get('profit', 0)
                if profit > 0:
                    winning_streak += 1
                    losing_streak = 0
                    max_winning_streak = max(max_winning_streak, winning_streak)
                else:
                    losing_streak += 1
                    winning_streak = 0
                    max_losing_streak = max(max_losing_streak, losing_streak)
            
            risk_analysis['consistency'] = max_winning_streak
            risk_analysis['drawdown_analysis'] = {
                'max_winning_streak': max_winning_streak,
                'max_losing_streak': max_losing_streak
            }
            
            return risk_analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحليل المخاطر: {e}")
            return risk_analysis
    
    def _generate_betting_recommendations(self, analysis: Dict) -> List[str]:
        """توليد توصيات الرهان بناءً على التحليل"""
        
        recommendations = []
        
        try:
            market_performance = analysis.get('by_market', {})
            
            # تحليل أفضل أنواع الرهان
            best_market = None
            best_roi = -100
            
            for market, performance in market_performance.items():
                roi = performance.get('roi', 0)
                if roi > best_roi:
                    best_roi = roi
                    best_market = market
            
            if best_market and best_roi > 5:
                recommendations.append(f"🎯 ركز على رهانات {best_market} (ROI: {best_roi:.1f}%)")
            
            # توصيات إدارة المخاطر
            risk_analysis = analysis.get('risk_analysis', {})
            max_losing_streak = risk_analysis.get('drawdown_analysis', {}).get('max_losing_streak', 0)
            
            if max_losing_streak > 5:
                recommendations.append(f"⚠️ تحسين إدارة المخاطر (أطول سلسلة خسائر: {max_losing_streak})")
            
            # توصيات عامة
            overall_perf = analysis.get('overall', {})
            accuracy = overall_perf.get('accuracy', 0)
            
            if accuracy < 0.5:
                recommendations.append("🔧 تحسين نماذج التنبؤ لزيادة الدقة")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في توليد التوصيات: {e}")
            return ["تحتاج إلى مزيد من البيانات لتوليد توصيات دقيقة"]
    
    def _evaluate_model_performance(self, seasonal_results: Dict) -> Dict:
        """تقييم أداء النماذج المختلفة"""
        
        model_evaluation = {
            'accuracy_analysis': {},
            'consistency_analysis': {},
            'improvement_recommendations': []
        }
        
        try:
            # جمع دقة النماذج عبر المواسم
            accuracy_by_season = {}
            
            for season_type, results in seasonal_results.items():
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    if 'model_accuracy' in metrics:
                        acc = metrics['model_accuracy']
                        accuracy_by_season[season_type] = {
                            'score_accuracy': acc.get('score_accuracy', 0),
                            'result_accuracy': acc.get('result_accuracy', 0)
                        }
            
            model_evaluation['accuracy_analysis'] = accuracy_by_season
            
            # تحليل الاتساق
            accuracies = [acc['result_accuracy'] for acc in accuracy_by_season.values()]
            if accuracies:
                model_evaluation['consistency_analysis'] = {
                    'average_accuracy': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'min_accuracy': min(accuracies),
                    'max_accuracy': max(accuracies)
                }
            
            # توليد توصيات التحسين
            avg_accuracy = model_evaluation['consistency_analysis'].get('average_accuracy', 0)
            
            if avg_accuracy < 0.4:
                model_evaluation['improvement_recommendations'].append("🔄 إعادة تدريب النماذج مع بيانات أحدث")
            if avg_accuracy < 0.5:
                model_evaluation['improvement_recommendations'].append("📊 إضافة ميزات تحليلية جديدة")
            if np.std(accuracies) > 0.1:
                model_evaluation['improvement_recommendations'].append("⚖️ تحسين اتساق النماذج عبر المواسم")
            
            return model_evaluation
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تقييم أداء النماذج: {e}")
            return model_evaluation
    
    def _simulate_future_betting(self, prediction_data: Dict, current_bankroll: float) -> Optional[Dict]:
        """محاكاة الرهان على المباريات المستقبلية"""
        
        try:
            stake = current_bankroll * 0.01  # 1% من الرصيد
            
            comprehensive_pred = prediction_data.get('comprehensive_prediction', {})
            betting_predictions = comprehensive_pred.get('betting_predictions', {})
            
            if not betting_predictions:
                return None
            
            # اختيار أفضل رهان بناءً على القيمة المتوقعة
            best_bet = None
            best_value = 0
            
            # تحليل رهانات 1X2
            if '1x2_market' in betting_predictions:
                for outcome, data in betting_predictions['1x2_market'].items():
                    prob = data.get('probability', 0)
                    odds = data.get('odds', 1.0)
                    
                    expected_value = (prob * (odds - 1)) - (1 - prob)
                    
                    if expected_value > best_value and expected_value > 0.05:
                        best_value = expected_value
                        best_bet = {
                            'market': '1x2',
                            'selection': outcome,
                            'probability': prob,
                            'odds': odds,
                            'expected_value': expected_value
                        }
            
            if best_bet:
                expected_profit = stake * best_value
                
                return {
                    'betting_decision': best_bet,
                    'stake': stake,
                    'odds': best_bet['odds'],
                    'expected_profit': expected_profit,
                    'net_profit': expected_profit  # للرسم البياني
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في محاكاة الرهان المستقبلي: {e}")
            return None
    
    def _calculate_future_performance_metrics(self, betting_results: List,
                                           initial_bankroll: float, final_bankroll: float) -> Dict:
        """حساب مقاييس الأداء للتنبؤات المستقبلية"""
        
        try:
            total_expected_profit = sum(b.get('expected_profit', 0) for b in betting_results)
            total_stake = sum(b.get('stake', 0) for b in betting_results)
            
            expected_roi = (total_expected_profit / total_stake) * 100 if total_stake > 0 else 0
            expected_growth = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
            
            return {
                'expected_performance': {
                    'total_expected_profit': total_expected_profit,
                    'expected_roi': expected_roi,
                    'expected_growth': expected_growth,
                    'total_bets': len(betting_results),
                    'average_expected_value': np.mean([b.get('expected_profit', 0) / b.get('stake', 1) 
                                                     for b in betting_results if b.get('stake', 0) > 0])
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حساب مقاييس الأداء المستقبلية: {e}")
            return {}
    
    def _save_comprehensive_backtest_results(self, results: Dict):
        """حفظ نتائج الـ Backtest الشامل"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # حفظ النتائج الرئيسية
            results_filename = f"backtest/results/comprehensive_backtest_{timestamp}.json"
            self._save_json(results, results_filename)
            
            # حفظ تحليل الرهان
            betting_analysis = results.get('betting_analysis', {})
            betting_filename = f"backtest/betting_analysis/betting_analysis_{timestamp}.json"
            self._save_json(betting_analysis, betting_filename)
            
            # حفظ تقرير الأداء
            performance_report = self._generate_performance_report(results)
            performance_filename = f"backtest/performance_reports/performance_report_{timestamp}.json"
            self._save_json(performance_report, performance_filename)
            
            # حفظ تحليل الموسم
            seasonal_analysis = results.get('seasonal_results', {})
            seasonal_filename = f"backtest/season_analysis/seasonal_analysis_{timestamp}.json"
            self._save_json(seasonal_analysis, seasonal_filename)
            
            self.logger.info(f"💾 تم حفظ نتائج الـ Backtest في: {results_filename}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ نتائج الـ Backtest: {e}")
    
    def _save_json(self, data: Dict, filepath: str):
        """حفظ البيانات في ملف JSON"""
        try:
            # إنشاء المجلد إذا لم يكن موجوداً
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ الملف {filepath}: {e}")
    
    def _generate_performance_report(self, results: Dict) -> Dict:
        """توليد تقرير أداء مفصل"""
        
        report = {
            'report_id': f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'executive_summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        try:
            overall_perf = results.get('overall_performance', {})
            betting_analysis = results.get('betting_analysis', {})
            model_perf = results.get('model_performance', {})
            
            # ملخص تنفيذي
            report['executive_summary'] = {
                'overall_accuracy': overall_perf.get('overall_accuracy', 0),
                'overall_roi': overall_perf.get('overall_roi', 0),
                'performance_rating': overall_perf.get('performance_rating', 'غير معروف'),
                'total_bets_placed': overall_perf.get('total_bets_placed', 0),
                'key_strengths': self._identify_strengths(overall_perf, betting_analysis),
                'key_improvements': self._identify_improvements(overall_perf, betting_analysis)
            }
            
            # تحليل مفصل
            report['detailed_analysis'] = {
                'model_performance': model_perf,
                'betting_performance': betting_analysis,
                'risk_analysis': betting_analysis.get('risk_analysis', {})
            }
            
            # التوصيات
            report['recommendations'] = self._generate_comprehensive_recommendations(results)
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد تقرير الأداء: {e}")
            return report
    
    def _identify_strengths(self, overall_perf: Dict, betting_analysis: Dict) -> List[str]:
        """تحديد نقاط القوة"""
        
        strengths = []
        
        if overall_perf.get('overall_accuracy', 0) > 0.55:
            strengths.append("دقة تنبؤ عالية")
        
        if overall_perf.get('overall_roi', 0) > 10:
            strengths.append("ربحية ممتازة")
        
        market_perf = betting_analysis.get('by_market', {})
        for market, perf in market_perf.items():
            if perf.get('roi', 0) > 15:
                strengths.append(f"أداء قوي في رهانات {market}")
        
        return strengths if strengths else ["أداء متوازن في معظم المجالات"]
    
    def _identify_improvements(self, overall_perf: Dict, betting_analysis: Dict) -> List[str]:
        """تحديد مجالات التحسين"""
        
        improvements = []
        
        if overall_perf.get('overall_accuracy', 0) < 0.45:
            improvements.append("تحسين دقة النماذج التنبؤية")
        
        risk_analysis = betting_analysis.get('risk_analysis', {})
        if risk_analysis.get('max_losing_streak', 0) > 8:
            improvements.append("تحسين إدارة المخاطر خلال فترات الخسارة")
        
        market_perf = betting_analysis.get('by_market', {})
        weak_markets = [market for market, perf in market_perf.items() 
                       if perf.get('roi', 0) < 0]
        
        if weak_markets:
            improvements.append(f"مراجعة استراتيجية رهانات {', '.join(weak_markets)}")
        
        return improvements if improvements else ["الأداء جيد بشكل عام"]
    
    def _generate_comprehensive_recommendations(self, results: Dict) -> List[str]:
        """توليد توصيات شاملة"""
        
        recommendations = []
        
        overall_perf = results.get('overall_performance', {})
        betting_analysis = results.get('betting_analysis', {})
        model_perf = results.get('model_performance', {})
        
        # توصيات بناءً على الأداء العام
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
        
        # توصيات بناءً على تحليل المخاطر
        risk_analysis = betting_analysis.get('risk_analysis', {})
        if risk_analysis.get('volatility', 0) > 500:
            recommendations.append("🛡️ تطبيق استراتيجية إدارة مخاطر أكثر تحفظاً")
        
        return recommendations

    def run_quick_backtest(self, matches_count: int = 100, initial_bankroll: float = 1000.0) -> Dict:
        """تشغيل Backtest سريع للتحقق من الأداء"""
        
        self.logger.info(f"⚡ بدء Backtest سريع لـ {matches_count} مباراة")
        
        quick_results = {
            'backtest_type': 'quick',
            'matches_analyzed': matches_count,
            'initial_bankroll': initial_bankroll,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # تحميل عينة سريعة من البيانات
            sample_data = self._load_historical_data_for_backtest("2023-2024", "2023-2024")
            sample_data = sample_data[:matches_count]
            
            if not sample_data:
                quick_results['error'] = "لا توجد بيانات للتحليل"
                return quick_results
            
            # محاكاة سريعة
            current_bankroll = initial_bankroll
            results = []
            
            for match_data in sample_data:
                try:
                    prediction = self._generate_prediction_for_backtest(match_data)
                    
                    if prediction:
                        bet_result = self._simulate_betting_for_match(
                            prediction, match_data, current_bankroll
                        )
                        
                        if bet_result:
                            current_bankroll += bet_result['net_profit']
                            results.append({
                                'match': f"{match_data.get('HomeTeam')} vs {match_data.get('AwayTeam')}",
                                'profit': bet_result['profit'],
                                'bankroll': current_bankroll
                            })
                
                except Exception as e:
                    continue
            
            # حساب النتائج
            total_profit = sum(r['profit'] for r in results)
            final_bankroll = current_bankroll
            winning_bets = len([r for r in results if r['profit'] > 0])
            total_bets = len(results)
            
            quick_results.update({
                'final_bankroll': final_bankroll,
                'total_profit': total_profit,
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'win_rate': winning_bets / total_bets if total_bets > 0 else 0,
                'roi': (total_profit / initial_bankroll) * 100 if initial_bankroll > 0 else 0,
                'success': True
            })
            
            self.logger.info("✅ اكتمل Backtest السريع")
            return quick_results
            
        except Exception as e:
            self.logger.error(f"❌ فشل في Backtest السريع: {e}")
            quick_results['error'] = str(e)
            quick_results['success'] = False
            return quick_results
        

# ==================== التشغيل الرئيسي لـ Backtester ====================

def main():
    """الدالة الرئيسية لتشغيل Backtester بشكل منفصل"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from advanced_system import EnhancedBettingSystem2025
    from output.output_manager import OutputManager
    
    print("🚀 بدء تشغيل نظام Backtest المتكامل...")
    
    # تهيئة النظام الرئيسي ومدير الإخراج
    output_manager = OutputManager()
    system = EnhancedBettingSystem2025("data/football-data")
    
    # تدريب النظام إذا لم يكن مدرباً
    if not system.is_trained:
        print("🔧 النظام غير مدرب، جاري التدريب...")
        system.train_with_temporal_integration(training_episodes=100)
    
    # تهيئة الـ Backtester
    backtester = AdvancedBacktester(system, output_manager)
    
    # خيارات التشغيل
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            # تشغيل Backtest سريع
            matches_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            results = backtester.run_quick_backtest(matches_count=matches_count)
        elif sys.argv[1] == "comprehensive":
            # تشغيل Backtest شامل
            start_season = sys.argv[2] if len(sys.argv) > 2 else "2020-2021"
            end_season = sys.argv[3] if len(sys.argv) > 3 else "2025-2026"
            bankroll = float(sys.argv[4]) if len(sys.argv) > 4 else 10000.0
            results = backtester.run_comprehensive_backtest(
                start_season=start_season,
                end_season=end_season,
                initial_bankroll=bankroll
            )
        else:
            print("❌ استخدام غير صحيح. الاختيارات المتاحة:")
            print("  python -m backtest.backtester quick [عدد_المباريات]")
            print("  python -m backtest.backtester comprehensive [بداية_الموسم] [نهاية_الموسم] [رأس_المال]")
            sys.exit(1)
    else:
        # التشغيل الافتراضي: Backtest سريع
        results = backtester.run_quick_backtest(matches_count=100)
    
    # عرض النتائج
    if results.get('success'):
        print("✅ Backtest اكتمل بنجاح!")
        
        if 'final_bankroll' in results:
            # نتائج Backtest سريع
            initial = results.get('initial_bankroll', 0)
            final = results.get('final_bankroll', 0)
            profit = results.get('total_profit', 0)
            roi = results.get('roi', 0)
            win_rate = results.get('win_rate', 0)
            
            print(f"📊 نتائج Backtest السريع:")
            print(f"   رأس المال الابتدائي: {initial:.2f} £")
            print(f"   رأس المال النهائي: {final:.2f} £")
            print(f"   إجمالي الربح: {profit:.2f} £")
            print(f"   العائد على الاستثمار: {roi:.2f}%")
            print(f"   معدل الفوز: {win_rate:.1%}")
            print(f"   عدد الرهانات: {results.get('total_bets', 0)}")
        
        elif 'overall_performance' in results:
            # نتائج Backtest شامل
            overall = results['overall_performance']
            print(f"📊 نتائج Backtest الشامل:")
            print(f"   الدقة العامة: {overall.get('overall_accuracy', 0):.1%}")
            print(f"   العائد على الاستثمار: {overall.get('overall_roi', 0):.1f}%")
            print(f"   معدل الفوز: {overall.get('overall_win_rate', 0):.1%}")
            print(f"   التقييم: {overall.get('performance_rating', 'غير معروف')}")
            print(f"   إجمالي الرهانات: {overall.get('total_bets_placed', 0)}")
            
            # عرض أفضل الأسواق أداءً
            betting_analysis = results.get('betting_analysis', {})
            markets = betting_analysis.get('by_market', {})
            if markets:
                print(f"   أفضل الأسواق أداءً:")
                for market, perf in list(markets.items())[:3]:
                    roi = perf.get('roi', 0)
                    if roi > 0:
                        print(f"     - {market}: {roi:.1f}% ROI")
        
        # حفظ النتائج في ملف منفصل
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest/results/backtest_execution_{timestamp}.json"
        backtester._save_json(results, results_file)
        print(f"💾 تم حفظ النتائج في: {results_file}")
        
    else:
        print(f"❌ فشل في تنفيذ Backtest: {results.get('error', 'خطأ غير معروف')}")

if __name__ == "__main__":
    main()