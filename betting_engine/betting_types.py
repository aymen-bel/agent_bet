# betting_engine/betting_types.py - النسخة المحسنة مع التكامل الكامل
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import logging

# ==================== أنواع الرهان ====================

class BetType(Enum):
    """أنواع الرهان المتاحة"""
    SINGLE = auto()
    ACCUMULATOR = auto()
    SYSTEM = auto()

class MarketType(Enum):
    """أنواع الأسواق للرهان"""
    RESULT = auto()           # 1X2
    DOUBLE_CHANCE = auto()    # 1X, X2, 12
    GOALS = auto()           # Over/Under
    SCORE = auto()           # Correct Score
    BTTS = auto()            # Both Teams To Score

# ==================== هياكل البيانات ====================

@dataclass
class Bet:
    """هيكل بيانات الرهان الأساسي"""
    bet_id: str
    bet_type: BetType
    market_type: MarketType
    selection: str
    odds: float
    stake: float
    confidence: float
    timestamp: datetime
    match_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل الرهان إلى قاموس"""
        return {
            'bet_id': self.bet_id,
            'bet_type': self.bet_type.name,
            'market_type': self.market_type.name,
            'selection': self.selection,
            'odds': float(self.odds),
            'stake': float(self.stake),
            'confidence': float(self.confidence),
            'timestamp': self.timestamp.isoformat(),
            'match_info': self.match_info,
            'potential_win': float(self.stake * self.odds),
            'expected_value': float((self.stake * self.odds * self.confidence) - (self.stake * (1 - self.confidence)))
        }

@dataclass
class BettingResult:
    """نتيجة الرهان"""
    bet_id: str
    is_winner: bool
    actual_odds: Optional[float] = None
    profit_loss: float = 0.0
    settlement_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل النتيجة إلى قاموس"""
        return {
            'bet_id': self.bet_id,
            'is_winner': self.is_winner,
            'actual_odds': float(self.actual_odds) if self.actual_odds else None,
            'profit_loss': float(self.profit_loss),
            'settlement_time': self.settlement_time.isoformat() if self.settlement_time else None
        }

# ==================== محرك الرهان المتكامل ====================

class BettingEngine:
    """محرك رهان متكامل مع جميع أنواع الرهان"""
    
    def __init__(self):
        self.bet_history: List[Bet] = []
        self.result_history: List[BettingResult] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        
        # إعدادات الرهان
        self.min_confidence = 0.6
        self.max_stake_per_bet = 100.0
        self.bankroll = 1000.0
        
    def _setup_logging(self) -> logging.Logger:
        """إعداد نظام التسجيل"""
        logger = logging.getLogger('BettingEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_odds(self, prediction: Dict, market_type: MarketType, **kwargs) -> Dict[str, float]:
        """توليد odds بناءً على التنبؤ ونوع السوق"""
        try:
            home_goals = prediction.get('home_goals', 1)
            away_goals = prediction.get('away_goals', 1)
            confidence = prediction.get('confidence', 0.5)
            
            if market_type == MarketType.RESULT:
                return self._generate_1x2_odds(home_goals, away_goals, confidence)
            elif market_type == MarketType.DOUBLE_CHANCE:
                return self._generate_double_chance_odds(home_goals, away_goals, confidence)
            elif market_type == MarketType.GOALS:
                line = kwargs.get('line', 2.5)
                return self._generate_over_under_odds(home_goals, away_goals, confidence, line)
            elif market_type == MarketType.SCORE:
                return self._generate_correct_score_odds(home_goals, away_goals, confidence)
            elif market_type == MarketType.BTTS:
                return self._generate_btts_odds(home_goals, away_goals, confidence)
            else:
                return self._get_fallback_odds(market_type)
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds: {e}")
            return self._get_fallback_odds(market_type)
    
    def _generate_1x2_odds(self, home_goals: int, away_goals: int, confidence: float) -> Dict[str, float]:
        """توليد odds لسوق 1X2"""
        try:
            # حساب الاحتمالات الأساسية
            total_goals = home_goals + away_goals
            home_win_prob = 0.33
            draw_prob = 0.33
            away_win_prob = 0.34
            
            if total_goals > 0:
                home_win_prob = home_goals / total_goals * 1.1
                away_win_prob = away_goals / total_goals * 0.9
                draw_prob = 1.0 - home_win_prob - away_win_prob
            
            # تطبيع الاحتمالات
            total_prob = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
            
            # تطبيق عامل الثقة
            confidence_boost = max(0.1, confidence - 0.5)
            if home_goals > away_goals:
                home_win_prob += confidence_boost
            elif away_goals > home_goals:
                away_win_prob += confidence_boost
            else:
                draw_prob += confidence_boost
            
            # إعادة التطبيع
            total_prob = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
            
            # حساب الـ odds مع هامش الربح
            margin = 0.05  # هامش 5%
            home_odds = round((1.0 / home_win_prob) * (1 - margin), 2)
            draw_odds = round((1.0 / draw_prob) * (1 - margin), 2)
            away_odds = round((1.0 / away_win_prob) * (1 - margin), 2)
            
            return {
                '1': max(1.5, home_odds),
                'X': max(2.0, draw_odds),
                '2': max(1.5, away_odds)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds 1X2: {e}")
            return {'1': 2.0, 'X': 3.0, '2': 2.5}
    
    def _generate_double_chance_odds(self, home_goals: int, away_goals: int, confidence: float) -> Dict[str, float]:
        """توليد odds للفرصة المزدوجة"""
        try:
            # الحصول على odds الأساسية
            base_odds = self._generate_1x2_odds(home_goals, away_goals, confidence)
            
            # حساب odds الفرصة المزدوجة
            odds_1x = 1.0 / ((1.0 / base_odds['1']) + (1.0 / base_odds['X']))
            odds_x2 = 1.0 / ((1.0 / base_odds['X']) + (1.0 / base_odds['2']))
            odds_12 = 1.0 / ((1.0 / base_odds['1']) + (1.0 / base_odds['2']))
            
            return {
                '1X': round(max(1.2, odds_1x), 2),
                'X2': round(max(1.2, odds_x2), 2),
                '12': round(max(1.1, odds_12), 2)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds الفرصة المزدوجة: {e}")
            return {'1X': 1.5, 'X2': 1.5, '12': 1.2}
    
    def _generate_over_under_odds(self, home_goals: int, away_goals: int, confidence: float, line: float = 2.5) -> Dict[str, float]:
        """توليد odds لـ Over/Under"""
        try:
            total_goals = home_goals + away_goals
            
            # حساب احتمالية Over/Under
            over_prob = 0.5
            under_prob = 0.5
            
            if total_goals > line:
                over_prob = min(0.9, 0.5 + (total_goals - line) * 0.1)
                under_prob = 1.0 - over_prob
            elif total_goals < line:
                under_prob = min(0.9, 0.5 + (line - total_goals) * 0.1)
                over_prob = 1.0 - under_prob
            
            # تطبيق عامل الثقة
            if confidence > 0.6:
                if total_goals > line:
                    over_prob += 0.1
                else:
                    under_prob += 0.1
            
            # تطبيع الاحتمالات
            total_prob = over_prob + under_prob
            over_prob /= total_prob
            under_prob /= total_prob
            
            # حساب الـ odds
            margin = 0.05
            over_odds = round((1.0 / over_prob) * (1 - margin), 2)
            under_odds = round((1.0 / under_prob) * (1 - margin), 2)
            
            return {
                'over': max(1.5, over_odds),
                'under': max(1.5, under_odds)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds Over/Under: {e}")
            return {'over': 2.0, 'under': 1.8}
    
    def _generate_correct_score_odds(self, home_goals: int, away_goals: int, confidence: float) -> Dict[str, float]:
        """توليد odds للنتيجة الصحيحة"""
        try:
            # التركيز على النتيجة المتوقعة
            predicted_score = f"{home_goals}-{away_goals}"
            
            # odds عالية للنتيجة الصحيحة مع تعديل بناءً على الثقة
            base_odds = 8.0
            confidence_adjustment = max(0.5, confidence)
            predicted_odds = round(base_odds / confidence_adjustment, 2)
            
            # إضافة بعض النتائج المحتملة الأخرى
            odds_dict = {predicted_score: predicted_odds}
            
            # نتائج محتملة قريبة
            similar_scores = [
                f"{max(0, home_goals-1)}-{away_goals}", f"{home_goals}-{max(0, away_goals-1)}",
                f"{home_goals+1}-{away_goals}", f"{home_goals}-{away_goals+1}",
                f"{max(0, home_goals-1)}-{max(0, away_goals-1)}", f"{home_goals+1}-{away_goals+1}"
            ]
            
            for score in similar_scores:
                if score != predicted_score:
                    odds_dict[score] = round(predicted_odds * 1.5, 2)
            
            return odds_dict
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds النتيجة الصحيحة: {e}")
            return {'1-1': 8.0, '2-1': 9.0, '1-2': 9.0, '2-0': 10.0, '0-2': 10.0}
    
    def _generate_btts_odds(self, home_goals: int, away_goals: int, confidence: float) -> Dict[str, float]:
        """توليد odds لـ Both Teams To Score"""
        try:
            # حساب احتمالية تسجيل كلا الفريقين
            btts_prob = 0.5
            
            if home_goals > 0 and away_goals > 0:
                btts_prob = min(0.9, 0.5 + (min(home_goals, away_goals) * 0.2))
            elif home_goals == 0 or away_goals == 0:
                btts_prob = max(0.1, 0.5 - (max(home_goals, away_goals) * 0.2))
            
            # تطبيق عامل الثقة
            if confidence > 0.6:
                if home_goals > 0 and away_goals > 0:
                    btts_prob += 0.15
                else:
                    btts_prob -= 0.15
            
            btts_prob = max(0.1, min(0.9, btts_prob))
            no_btts_prob = 1.0 - btts_prob
            
            # حساب الـ odds
            margin = 0.05
            btts_yes_odds = round((1.0 / btts_prob) * (1 - margin), 2)
            btts_no_odds = round((1.0 / no_btts_prob) * (1 - margin), 2)
            
            return {
                'yes': max(1.5, btts_yes_odds),
                'no': max(1.5, btts_no_odds)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds BTTS: {e}")
            return {'yes': 1.8, 'no': 1.9}
    
    def _get_fallback_odds(self, market_type: MarketType) -> Dict[str, float]:
        """إرجاع odds افتراضية في حالة الخطأ"""
        fallback_odds = {
            MarketType.RESULT: {'1': 2.0, 'X': 3.0, '2': 2.5},
            MarketType.DOUBLE_CHANCE: {'1X': 1.5, 'X2': 1.5, '12': 1.2},
            MarketType.GOALS: {'over': 2.0, 'under': 1.8},
            MarketType.SCORE: {'1-1': 8.0, '2-1': 9.0, '1-2': 9.0},
            MarketType.BTTS: {'yes': 1.8, 'no': 1.9}
        }
        return fallback_odds.get(market_type, {'default': 2.0})
    
    def create_single_bet(self, prediction: Dict, market_type: MarketType, 
                         selection: str, stake: float, match_info: Dict) -> Optional[Bet]:
        """إنشاء رهان فردي"""
        try:
            # التحقق من الحد الأدنى للثقة
            confidence = prediction.get('confidence', 0.0)
            if confidence < self.min_confidence:
                self.logger.warning(f"⚠️  ثقة منخفضة للرهان: {confidence:.2f}")
                return None
            
            # توليد الـ odds
            odds_dict = self.generate_odds(prediction, market_type)
            odds = odds_dict.get(selection)
            
            if not odds or odds < 1.1:
                self.logger.warning(f"⚠️  odds غير صالحة: {odds}")
                return None
            
            # التحقق من قيمة الرهان
            if stake > self.max_stake_per_bet:
                stake = self.max_stake_per_bet
                self.logger.info(f"📊 تعديل قيمة الرهان إلى الحد الأقصى: {stake}")
            
            # إنشاء الرهان
            bet_id = f"bet_{len(self.bet_history) + 1:06d}"
            bet = Bet(
                bet_id=bet_id,
                bet_type=BetType.SINGLE,
                market_type=market_type,
                selection=selection,
                odds=float(odds),
                stake=float(stake),
                confidence=float(confidence),
                timestamp=datetime.now(),
                match_info=match_info
            )
            
            self.bet_history.append(bet)
            self.logger.info(f"✅ تم إنشاء رهان فردي: {bet_id}")
            
            return bet
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء الرهان الفردي: {e}")
            return None
    
    def create_accumulator_bet(self, predictions_list: List[Dict], market_types: List[MarketType],
                              selections: List[str], stake: float, matches_info: List[Dict]) -> Optional[Bet]:
        """إنشاء رهان متراكم"""
        try:
            if len(predictions_list) != len(market_types) != len(selections) != len(matches_info):
                self.logger.error("❌ عدد التنبؤات والاختيارات والمباريات غير متطابق")
                return None
            
            # التحقق من الحد الأدنى للثقة لجميع المباريات
            min_confidence = min(pred.get('confidence', 0.0) for pred in predictions_list)
            if min_confidence < self.min_confidence:
                self.logger.warning(f"⚠️  ثقة منخفضة في الرهان المتراكم: {min_confidence:.2f}")
                return None
            
            # حساب الـ odds الإجمالية
            total_odds = 1.0
            for i, (pred, market_type, selection) in enumerate(zip(predictions_list, market_types, selections)):
                odds_dict = self.generate_odds(pred, market_type)
                odds = odds_dict.get(selection, 1.0)
                total_odds *= odds
            
            if total_odds <= 1.0:
                self.logger.warning(f"⚠️  odds إجمالية غير صالحة: {total_odds}")
                return None
            
            # إنشاء الرهان المتراكم
            bet_id = f"acc_{len(self.bet_history) + 1:06d}"
            bet = Bet(
                bet_id=bet_id,
                bet_type=BetType.ACCUMULATOR,
                market_type=market_types[0],  # استخدام أول نوع سوق كتمثيل
                selection=str(selections),
                odds=float(total_odds),
                stake=float(stake),
                confidence=float(min_confidence),
                timestamp=datetime.now(),
                match_info={'matches': matches_info, 'count': len(matches_info)}
            )
            
            self.bet_history.append(bet)
            self.logger.info(f"✅ تم إنشاء رهان متراكم: {bet_id} مع {len(matches_info)} مباراة")
            
            return bet
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء الرهان المتراكم: {e}")
            return None
    
    def create_system_bet(self, predictions_list: List[Dict], market_types: List[MarketType],
                         selections_list: List[List[str]], stake: float, matches_info: List[Dict]) -> List[Bet]:
        """إنشاء رهان نظام"""
        try:
            system_bets = []
            
            # إنشاء رهانات فردية للنظام
            for i, (pred, market_type, selections) in enumerate(zip(predictions_list, market_types, selections_list)):
                for selection in selections:
                    single_bet = self.create_single_bet(pred, market_type, selection, stake, matches_info[i])
                    if single_bet:
                        single_bet.bet_type = BetType.SYSTEM
                        system_bets.append(single_bet)
            
            self.logger.info(f"✅ تم إنشاء {len(system_bets)} رهان نظام")
            return system_bets
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء رهان النظام: {e}")
            return []
    
    def calculate_bet_metrics(self, bet: Bet) -> Dict[str, float]:
        """حساب مقاييس الرهان"""
        try:
            expected_value = (bet.stake * bet.odds * bet.confidence) - (bet.stake * (1 - bet.confidence))
            kelly_criterion = (bet.odds * bet.confidence - 1) / (bet.odds - 1) if bet.odds > 1 else 0.0
            kelly_criterion = max(0.0, min(0.1, kelly_criterion))  # تحديد بين 0% و 10%
            
            return {
                'expected_value': float(expected_value),
                'kelly_criterion': float(kelly_criterion),
                'potential_win': float(bet.stake * bet.odds),
                'potential_loss': float(bet.stake),
                'profit_probability': float(bet.confidence),
                'loss_probability': float(1 - bet.confidence)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حساب مقاييس الرهان: {e}")
            return {
                'expected_value': 0.0,
                'kelly_criterion': 0.0,
                'potential_win': 0.0,
                'potential_loss': 0.0,
                'profit_probability': 0.0,
                'loss_probability': 1.0
            }
    
    def settle_bet(self, bet_id: str, is_winner: bool, actual_odds: Optional[float] = None) -> Optional[BettingResult]:
        """تسوية الرهان"""
        try:
            # البحث عن الرهان
            bet = next((b for b in self.bet_history if b.bet_id == bet_id), None)
            if not bet:
                self.logger.error(f"❌ رهان غير موجود: {bet_id}")
                return None
            
            # حساب الربح/الخسارة
            if is_winner:
                profit_loss = (bet.stake * (actual_odds if actual_odds else bet.odds)) - bet.stake
            else:
                profit_loss = -bet.stake
            
            # تحديث الرصيد
            self.bankroll += profit_loss
            
            # إنشاء نتيجة الرهان
            result = BettingResult(
                bet_id=bet_id,
                is_winner=is_winner,
                actual_odds=float(actual_odds) if actual_odds else None,
                profit_loss=float(profit_loss),
                settlement_time=datetime.now()
            )
            
            self.result_history.append(result)
            self.logger.info(f"✅ تم تسوية الرهان {bet_id}: {'ربح' if is_winner else 'خسارة'} ({profit_loss:+.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تسوية الرهان: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """الحصول على مقاييس أداء الرهان"""
        try:
            if not self.result_history:
                return {
                    'total_bets': 0,
                    'winning_bets': 0,
                    'losing_bets': 0,
                    'win_rate': 0.0,
                    'total_profit_loss': 0.0,
                    'average_profit_per_bet': 0.0,
                    'roi': 0.0,
                    'bankroll': float(self.bankroll)
                }
            
            total_bets = len(self.result_history)
            winning_bets = len([r for r in self.result_history if r.is_winner])
            losing_bets = total_bets - winning_bets
            total_profit_loss = sum(r.profit_loss for r in self.result_history)
            total_stake = sum(b.stake for b in self.bet_history if b.bet_id in [r.bet_id for r in self.result_history])
            
            win_rate = winning_bets / total_bets if total_bets > 0 else 0.0
            avg_profit = total_profit_loss / total_bets if total_bets > 0 else 0.0
            roi = (total_profit_loss / total_stake) * 100 if total_stake > 0 else 0.0
            
            return {
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'losing_bets': losing_bets,
                'win_rate': float(win_rate),
                'total_profit_loss': float(total_profit_loss),
                'average_profit_per_bet': float(avg_profit),
                'roi': float(roi),
                'bankroll': float(self.bankroll),
                'performance_rating': self._calculate_performance_rating(win_rate, roi)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حساب مقاييس الأداء: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_rating(self, win_rate: float, roi: float) -> str:
        """حساب تقييم الأداء"""
        if win_rate > 0.6 and roi > 15:
            return "ممتاز"
        elif win_rate > 0.55 and roi > 10:
            return "جيد جداً"
        elif win_rate > 0.5 and roi > 5:
            return "جيد"
        elif win_rate > 0.45 and roi > 0:
            return "مقبول"
        else:
            return "ضعيف"
    
    def get_betting_recommendations(self, predictions: List[Dict], matches_info: List[Dict], 
                                  max_recommendations: int = 5) -> List[Dict]:
        """توليد توصيات الرهان"""
        try:
            recommendations = []
            
            for i, (pred, match_info) in enumerate(zip(predictions, matches_info)):
                confidence = pred.get('confidence', 0.0)
                home_goals = pred.get('home_goals', 1)
                away_goals = pred.get('away_goals', 1)
                
                # فقط المباريات ذات الثقة العالية
                if confidence < self.min_confidence:
                    continue
                
                # توليد توصيات بناءً على التنبؤ
                market_recommendations = self._generate_market_recommendations(
                    home_goals, away_goals, confidence, match_info
                )
                
                recommendations.extend(market_recommendations)
            
            # ترتيب التوصيات حسب القيمة المتوقعة
            recommendations.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
            
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد التوصيات: {e}")
            return []
    
    def _generate_market_recommendations(self, home_goals: int, away_goals: int, 
                                       confidence: float, match_info: Dict) -> List[Dict]:
        """توليد توصيات للأسواق المختلفة"""
        recommendations = []
        
        try:
            # توصيات سوق 1X2
            if home_goals > away_goals and confidence > 0.65:
                odds = self._generate_1x2_odds(home_goals, away_goals, confidence)
                recommendations.append({
                    'market': '1X2',
                    'selection': '1',
                    'odds': odds.get('1', 2.0),
                    'confidence': confidence,
                    'expected_value': (odds.get('1', 2.0) * confidence) - 1,
                    'reason': f'فوز متوقع للمنزل {home_goals}-{away_goals}'
                })
            elif away_goals > home_goals and confidence > 0.65:
                odds = self._generate_1x2_odds(home_goals, away_goals, confidence)
                recommendations.append({
                    'market': '1X2',
                    'selection': '2',
                    'odds': odds.get('2', 2.5),
                    'confidence': confidence,
                    'expected_value': (odds.get('2', 2.5) * confidence) - 1,
                    'reason': f'فوز متوقع للضيف {away_goals}-{home_goals}'
                })
            
            # توصيات Over/Under
            total_goals = home_goals + away_goals
            if total_goals > 3 and confidence > 0.6:
                odds = self._generate_over_under_odds(home_goals, away_goals, confidence, 2.5)
                recommendations.append({
                    'market': 'Over/Under',
                    'selection': 'over_2.5',
                    'odds': odds.get('over', 2.0),
                    'confidence': min(0.9, confidence + 0.1),
                    'expected_value': (odds.get('over', 2.0) * min(0.9, confidence + 0.1)) - 1,
                    'reason': f'أهداف متوقعة عالية: {total_goals}'
                })
            elif total_goals < 2 and confidence > 0.6:
                odds = self._generate_over_under_odds(home_goals, away_goals, confidence, 2.5)
                recommendations.append({
                    'market': 'Over/Under',
                    'selection': 'under_2.5',
                    'odds': odds.get('under', 1.8),
                    'confidence': min(0.9, confidence + 0.1),
                    'expected_value': (odds.get('under', 1.8) * min(0.9, confidence + 0.1)) - 1,
                    'reason': f'أهداف متوقعة منخفضة: {total_goals}'
                })
            
            # توصيات BTTS
            if home_goals > 0 and away_goals > 0 and confidence > 0.55:
                odds = self._generate_btts_odds(home_goals, away_goals, confidence)
                recommendations.append({
                    'market': 'BTTS',
                    'selection': 'yes',
                    'odds': odds.get('yes', 1.8),
                    'confidence': min(0.85, confidence + 0.15),
                    'expected_value': (odds.get('yes', 1.8) * min(0.85, confidence + 0.15)) - 1,
                    'reason': 'كلا الفريقين متوقع أن يسجلا'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد توصيات السوق: {e}")
            return []
    
    def export_betting_data(self, filepath: str) -> bool:
        """تصدير بيانات الرهان إلى ملف"""
        try:
            betting_data = {
                'export_timestamp': datetime.now().isoformat(),
                'bankroll': float(self.bankroll),
                'bet_history': [bet.to_dict() for bet in self.bet_history],
                'result_history': [result.to_dict() for result in self.result_history],
                'performance_metrics': self.get_performance_metrics()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(betting_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 تم تصدير بيانات الرهان إلى: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تصدير بيانات الرهان: {e}")
            return False
    
    def import_betting_data(self, filepath: str) -> bool:
        """استيراد بيانات الرهان من ملف"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                betting_data = json.load(f)
            
            # استيراد سجل الرهانات
            self.bet_history = []
            for bet_dict in betting_data.get('bet_history', []):
                try:
                    bet = Bet(
                        bet_id=bet_dict['bet_id'],
                        bet_type=BetType[bet_dict['bet_type']],
                        market_type=MarketType[bet_dict['market_type']],
                        selection=bet_dict['selection'],
                        odds=float(bet_dict['odds']),
                        stake=float(bet_dict['stake']),
                        confidence=float(bet_dict['confidence']),
                        timestamp=datetime.fromisoformat(bet_dict['timestamp']),
                        match_info=bet_dict['match_info']
                    )
                    self.bet_history.append(bet)
                except Exception as e:
                    self.logger.warning(f"⚠️  خطأ في استيراد رهان: {e}")
                    continue
            
            # استيراد سجل النتائج
            self.result_history = []
            for result_dict in betting_data.get('result_history', []):
                try:
                    result = BettingResult(
                        bet_id=result_dict['bet_id'],
                        is_winner=result_dict['is_winner'],
                        actual_odds=float(result_dict['actual_odds']) if result_dict.get('actual_odds') else None,
                        profit_loss=float(result_dict['profit_loss']),
                        settlement_time=datetime.fromisoformat(result_dict['settlement_time']) if result_dict.get('settlement_time') else None
                    )
                    self.result_history.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️  خطأ في استيراد نتيجة: {e}")
                    continue
            
            # استيراد الرصيد
            self.bankroll = float(betting_data.get('bankroll', 1000.0))
            
            self.logger.info(f"📥 تم استيراد بيانات الرهان من: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في استيراد بيانات الرهان: {e}")
            return False

# ==================== وظائف مساعدة ====================

def calculate_kelly_criterion(odds: float, probability: float) -> float:
    """حساب معيار كيلي لتحديد حجم الرهان الأمثل"""
    try:
        if odds <= 1.0 or probability <= 0.0:
            return 0.0
        
        kelly = (odds * probability - 1) / (odds - 1)
        return max(0.0, min(0.1, kelly))  # تحديد بين 0% و 10%
    
    except Exception as e:
        logging.error(f"❌ خطأ في حساب معيار كيلي: {e}")
        return 0.0

def calculate_expected_value(odds: float, probability: float, stake: float) -> float:
    """حساب القيمة المتوقعة للرهان"""
    try:
        win_amount = stake * (odds - 1)
        loss_amount = stake
        expected_value = (probability * win_amount) - ((1 - probability) * loss_amount)
        return expected_value
    
    except Exception as e:
        logging.error(f"❌ خطأ في حساب القيمة المتوقعة: {e}")
        return 0.0

def analyze_betting_strategy(predictions: List[Dict], matches_info: List[Dict], 
                           strategy_type: str = "conservative") -> Dict[str, Any]:
    """تحليل استراتيجية الرهان"""
    try:
        engine = BettingEngine()
        
        if strategy_type == "conservative":
            engine.min_confidence = 0.7
            engine.max_stake_per_bet = 50.0
        elif strategy_type == "moderate":
            engine.min_confidence = 0.6
            engine.max_stake_per_bet = 75.0
        elif strategy_type == "aggressive":
            engine.min_confidence = 0.55
            engine.max_stake_per_bet = 100.0
        
        # توليد توصيات
        recommendations = engine.get_betting_recommendations(predictions, matches_info)
        
        # محاكاة الأداء
        simulated_results = []
        total_investment = 0.0
        total_return = 0.0
        
        for rec in recommendations:
            stake = min(engine.max_stake_per_bet, 100.0 * calculate_kelly_criterion(rec['odds'], rec['confidence']))
            total_investment += stake
            
            # محاكاة النتيجة بناءً على الاحتمال
            is_win = np.random.random() < rec['confidence']
            return_amount = stake * rec['odds'] if is_win else 0.0
            total_return += return_amount
            
            simulated_results.append({
                'recommendation': rec,
                'stake': stake,
                'is_win': is_win,
                'return': return_amount,
                'profit_loss': return_amount - stake
            })
        
        total_profit = total_return - total_investment
        roi = (total_profit / total_investment) * 100 if total_investment > 0 else 0.0
        
        return {
            'strategy_type': strategy_type,
            'total_recommendations': len(recommendations),
            'total_investment': total_investment,
            'total_return': total_return,
            'total_profit': total_profit,
            'roi': roi,
            'simulated_results': simulated_results,
            'performance_rating': engine._calculate_performance_rating(
                len([r for r in simulated_results if r['is_win']]) / len(simulated_results) if simulated_results else 0,
                roi
            )
        }
        
    except Exception as e:
        logging.error(f"❌ خطأ في تحليل الاستراتيجية: {e}")
        return {'error': str(e)}

# ==================== مثال على الاستخدام ====================

def example_usage():
    """مثال على استخدام محرك الرهان"""
    print("🎰 مثال على استخدام محرك الرهان المتكامل")
    
    # إنشاء محرك الرهان
    engine = BettingEngine()
    
    # تنبؤ نموذجي
    sample_prediction = {
        'home_goals': 2,
        'away_goals': 1,
        'confidence': 0.75
    }
    
    match_info = {
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
        'date': '2025-01-15',
        'venue': 'Emirates Stadium'
    }
    
    # توليد odds لأنواع مختلفة
    print("\n📊 odds المتولدة:")
    
    odds_1x2 = engine.generate_odds(sample_prediction, MarketType.RESULT)
    print(f"🎯 1X2 Odds: {odds_1x2}")
    
    odds_double = engine.generate_odds(sample_prediction, MarketType.DOUBLE_CHANCE)
    print(f"🎯 Double Chance Odds: {odds_double}")
    
    odds_goals = engine.generate_odds(sample_prediction, MarketType.GOALS, line=2.5)
    print(f"🎯 Over/Under 2.5 Odds: {odds_goals}")
    
    odds_btts = engine.generate_odds(sample_prediction, MarketType.BTTS)
    print(f"🎯 BTTS Odds: {odds_btts}")
    
    # إنشاء رهان فردي
    single_bet = engine.create_single_bet(
        sample_prediction, MarketType.RESULT, '1', 50.0, match_info
    )
    
    if single_bet:
        print(f"\n✅ الرهان الفردي المنشأ:")
        print(f"   🆔 الرمز: {single_bet.bet_id}")
        print(f"   🎯 الاختيار: {single_bet.selection}")
        print(f"   📈 Odds: {single_bet.odds}")
        print(f"   💰 القيمة: {single_bet.stake}")
        print(f"   💪 الثقة: {single_bet.confidence:.1%}")
        
        # حساب المقاييس
        metrics = engine.calculate_bet_metrics(single_bet)
        print(f"   📊 القيمة المتوقعة: {metrics['expected_value']:.2f}")
        print(f"   🎯 معيار كيلي: {metrics['kelly_criterion']:.1%}")
    
    # توليد توصيات
    recommendations = engine.get_betting_recommendations([sample_prediction], [match_info])
    print(f"\n💡 التوصيات المتولدة: {len(recommendations)}")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['market']} - {rec['selection']} (Odds: {rec['odds']}, ثقة: {rec['confidence']:.1%})")
    
    # مقاييس الأداء
    performance = engine.get_performance_metrics()
    print(f"\n📈 مقاييس الأداء:")
    print(f"   🏦 الرصيد: {performance['bankroll']:.2f}")
    print(f"   📊 معدل الفوز: {performance['win_rate']:.1%}")
    print(f"   💰 إجمالي الربح/الخسارة: {performance['total_profit_loss']:.2f}")
    print(f"   📈 العائد على الاستثمار: {performance['roi']:.1f}%")
    print(f"   ⭐ التقييم: {performance['performance_rating']}")

if __name__ == "__main__":
    example_usage()