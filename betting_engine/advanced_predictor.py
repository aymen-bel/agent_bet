# betting_engine/advanced_predictor.py

import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from .betting_types import BettingEngine, BetType, MarketType

class AdvancedBettingPredictor:
    def __init__(self, main_system):
        self.main_system = main_system
        self.betting_engine = BettingEngine()
        self.logger = logging.getLogger(__name__)
        self.bet_history = []
    
    def generate_comprehensive_predictions(self, matches_data: List[Dict], stake: float = 100.0) -> Dict:
     """توليد تنبؤات شاملة للرهان"""
     try:
        bet_results = []
        
        for match_data in matches_data:
            # التأكد من وجود البيانات المطلوبة
            if 'team_metrics' not in match_data or 'opponent_metrics' not in match_data or 'context' not in match_data:
                continue
                
            home_team = match_data['context'].get('home_team', 'Unknown')
            away_team = match_data['context'].get('away_team', 'Unknown')
            
            # استخدام محرك الرهان الأساسي
            bet_result = self.betting_engine.calculate_single_bet(
                stake=stake,
                odds=2.0,  # يمكنك جلب odds حقيقية من مصدر خارجي
                prediction=match_data.get('prediction', {}),
                actual_result={}  # لن تكون هناك نتائج فعلية للمباريات المستقبلية
            )
            
            bet_results.append({
                'match_info': {
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': match_data['context'].get('match_date', 'Unknown')
                },
                'bet_result': bet_result
            })
        
        return {
            'bet_results': bet_results,
            'total_stake': stake * len(bet_results),
            'potential_profit': sum(r['bet_result'].get('profit', 0) for r in bet_results),
            'generated_at': datetime.now().isoformat()
        }
        
     except Exception as e:
        logging.error(f"❌ خطأ في توليد التنبؤات الشاملة: {e}")
        return {}
    
    def _generate_all_bet_types(self, prediction: Dict, match_data: Dict, stake: float) -> List[Dict]:
        """توليد جميع أنواع الرهانات لمباراة واحدة"""
        bet_results = []
        actual_result = match_data['actual_result']
        
        try:
            # Single Bet
            odds = self.betting_engine.generate_odds(prediction, MarketType.RESULT)
            single_bet = self.betting_engine.calculate_single_bet(
                stake, odds['1'], prediction, actual_result
            )
            bet_results.append(single_bet)
            
            # 1X2 Market
            market_odds = {'1': odds['1'], 'X': odds['X'], '2': odds['2']}
            bet_1x2 = self.betting_engine.calculate_1x2_bet(
                stake, prediction, actual_result, market_odds
            )
            bet_results.append(bet_1x2)
            
            # Double Chance
            dc_odds = {'1X': odds['1X'], 'X2': odds['X2']}
            bet_dc = self.betting_engine.calculate_double_chance_bet(
                stake, prediction, actual_result, dc_odds
            )
            bet_results.append(bet_dc)
            
            # Both Teams To Score
            btts_odds = 1.8  # odds افتراضية
            bet_btts = self.betting_engine.calculate_btts_bet(
                stake, prediction, actual_result, btts_odds
            )
            bet_results.append(bet_btts)
            
            # Over/Under
            ou_odds = {'over': 1.9, 'under': 1.9}
            bet_ou = self.betting_engine.calculate_over_under_bet(
                stake, prediction, actual_result, 2.5, ou_odds
            )
            bet_results.append(bet_ou)
            
            # Correct Score
            cs_odds = 8.0  # odds افتراضية للنتيجة الدقيقة
            bet_cs = self.betting_engine.calculate_correct_score_bet(
                stake, prediction, actual_result, cs_odds
            )
            bet_results.append(bet_cs)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد أنواع الرهانات: {e}")
        
        return bet_results
    
    def generate_accumulator_bets(self, matches_data: List[Dict], stake: float = 100.0) -> Dict:
        """توليد رهانات متراكمة"""
        try:
            predictions = []
            actual_results = []
            odds_list = []
            
            for match_data in matches_data:
                prediction = self.main_system.predict_match(
                    match_data['team_metrics'],
                    match_data['opponent_metrics'],
                    match_data['context']
                )
                
                predictions.append(prediction)
                actual_results.append(match_data['actual_result'])
                
                # توليد odds
                odds = self.betting_engine.generate_odds(prediction, MarketType.RESULT)
                odds_list.append(odds['1'])  # استخدام odds الفوز للفريق الأول
            
            # Accumulator Bet
            acc_bet = self.betting_engine.calculate_accumulator_bet(
                stake, predictions, actual_results, odds_list
            )
            
            # System Bet (2 من 3)
            if len(predictions) >= 3:
                system_bet = self.betting_engine.calculate_system_bet(
                    stake, predictions[:3], actual_results[:3], odds_list[:3], "2_3"
                )
            else:
                system_bet = {'profit': 0, 'accuracy': 0}
            
            return {
                'accumulator': acc_bet,
                'system': system_bet,
                'total_matches': len(matches_data)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد الرهانات المتراكمة: {e}")
            return {}
    
    def get_betting_recommendations(self, matches_data: List[Dict]) -> Dict:
        """توليد توصيات رهانات ذكية"""
        try:
            recommendations = {
                'single_bets': [],
                'accumulator_opportunities': [],
                'high_confidence_bets': [],
                'value_bets': [],
                'risk_analysis': {}
            }
            
            for match_data in matches_data:
                prediction = self.main_system.predict_match(
                    match_data['team_metrics'],
                    match_data['opponent_metrics'],
                    match_data['context']
                )
                
                confidence = prediction.get('model_confidence', 0.5)
                
                # تحليل القيمة
                odds = self.betting_engine.generate_odds(prediction, MarketType.RESULT)
                implied_prob = 1.0 / odds['1']  # احتمالية ضمنية من الـ odds
                
                # حساب قيمة الرهان
                if confidence > 0.7:
                    recommendations['high_confidence_bets'].append({
                        'match': f"{match_data['context'].get('home_team')} vs {match_data['context'].get('away_team')}",
                        'confidence': confidence,
                        'recommended_bet': '1X2 Market',
                        'suggested_odds': odds
                    })
                
                if confidence > implied_prob + 0.1:  # قيمة إيجابية
                    recommendations['value_bets'].append({
                        'match': f"{match_data['context'].get('home_team')} vs {match_data['context'].get('away_team')}",
                        'value_score': confidence - implied_prob,
                        'recommended_odds': odds['1']
                    })
            
            # تحليل المخاطر
            total_confidence = sum(bet['confidence'] for bet in recommendations['high_confidence_bets'])
            avg_confidence = total_confidence / len(recommendations['high_confidence_bets']) if recommendations['high_confidence_bets'] else 0
            
            recommendations['risk_analysis'] = {
                'total_recommended_bets': len(recommendations['high_confidence_bets']),
                'average_confidence': avg_confidence,
                'risk_level': 'LOW' if avg_confidence > 0.7 else 'MEDIUM' if avg_confidence > 0.5 else 'HIGH'
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد التوصيات: {e}")
            return {}