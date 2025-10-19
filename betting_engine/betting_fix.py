# betting_fix.py
import sys
import os
import pandas as pd
from typing import Dict, List, Optional
import json

class BettingDataFixer:
    def __init__(self):
        self.required_context_fields = ['home_team', 'away_team', 'match_date', 'week_number']
    
    def fix_match_data_for_betting(self, matches_data: List[Dict]) -> List[Dict]:
        """إصلاح بيانات المباريات لتتوافق مع متطلبات محرك الرهان"""
        fixed_matches = []
        
        for match in matches_data:
            try:
                fixed_match = self._fix_single_match(match)
                if fixed_match:
                    fixed_matches.append(fixed_match)
            except Exception as e:
                print(f"⚠️ خطأ في إصلاح المباراة: {e}")
                continue
        
        return fixed_matches
    
    def _fix_single_match(self, match: Dict) -> Optional[Dict]:
        """إصلاح مباراة فردية"""
        # إنشاء context إذا كان مفقوداً
        if 'context' not in match:
            match['context'] = {}
        
        # تعبئة الحقول المطلوبة في context
        context = match['context']
        
        # الحصول على أسماء الفرق من مصادر مختلفة
        home_team = (context.get('home_team') or 
                    match.get('home_team') or 
                    match.get('match_info', {}).get('home_team', 'Unknown'))
        
        away_team = (context.get('away_team') or 
                    match.get('away_team') or 
                    match.get('match_info', {}).get('away_team', 'Unknown'))
        
        context['home_team'] = home_team
        context['away_team'] = away_team
        context['match_date'] = context.get('match_date') or match.get('match_date', 'Unknown')
        context['week_number'] = context.get('week_number') or match.get('week_number', 0)
        context['is_current_season'] = True
        
        # إنشاء team_metrics إذا كانت مفقودة
        if 'team_metrics' not in match:
            match['team_metrics'] = self._create_default_metrics(home_team, is_home=True)
        
        if 'opponent_metrics' not in match:
            match['opponent_metrics'] = self._create_default_metrics(away_team, is_home=False)
        
        # إضافة تنبؤ أساسي إذا كان مفقوداً
        if 'prediction' not in match:
            match['prediction'] = self._create_basic_prediction(match)
        
        return match
    
    def _create_default_metrics(self, team_name: str, is_home: bool) -> Dict:
        """إنشاء مقاييس افتراضية للفريق"""
        return {
            'points_per_match': 1.5,
            'win_rate': 0.5,
            'goals_per_match': 1.2,
            'conceded_per_match': 1.2,
            'current_form': 0.5,
            'home_advantage': 1.15 if is_home else 0.85,
            'defensive_efficiency': 0.7,
            'motivation_factor': 1.0,
            'team_strength': 1.0,
            'attack_strength': 1.0,
            'defense_strength': 1.0
        }
    
    def _create_basic_prediction(self, match: Dict) -> Dict:
        """إنشاء تنبؤ أساسي"""
        return {
            'home_goals': 1,
            'away_goals': 1,
            'confidence': 0.5,
            'prediction_type': 'fallback'
        }

class SimpleBettingEngine:
    """محرك رهان مبسط بديل"""
    
    def __init__(self):
        self.bet_history = []
    
    def calculate_simple_bet(self, match_data: Dict, stake: float = 100.0) -> Dict:
        """حساب رهان مبسط"""
        try:
            context = match_data.get('context', {})
            prediction = match_data.get('prediction', {})
            
            home_team = context.get('home_team', 'Unknown')
            away_team = context.get('away_team', 'Unknown')
            
            home_goals = prediction.get('home_goals', 1)
            away_goals = prediction.get('away_goals', 1)
            confidence = prediction.get('confidence', 0.5)
            
            # حساب odds مبسطة بناءً على التنبؤ
            if home_goals > away_goals:
                odds = 2.0 + (confidence * 1.0)
                bet_type = "HOME_WIN"
            elif away_goals > home_goals:
                odds = 2.5 + (confidence * 1.0)
                bet_type = "AWAY_WIN"
            else:
                odds = 3.0 + (confidence * 1.0)
                bet_type = "DRAW"
            
            # حساب الربح المحتمل
            potential_profit = stake * (odds - 1)
            
            bet_result = {
                'match': f"{home_team} vs {away_team}",
                'prediction': f"{home_goals}-{away_goals}",
                'bet_type': bet_type,
                'stake': stake,
                'odds': round(odds, 2),
                'potential_profit': round(potential_profit, 2),
                'confidence': confidence,
                'recommendation': self._generate_recommendation(confidence, potential_profit)
            }
            
            self.bet_history.append(bet_result)
            return bet_result
            
        except Exception as e:
            print(f"❌ خطأ في حساب الرهان: {e}")
            return {
                'error': str(e),
                'stake': stake,
                'potential_profit': -stake
            }
    
    def _generate_recommendation(self, confidence: float, profit: float) -> str:
        """توليد توصية بناءً على الثقة والربح"""
        if confidence >= 0.7 and profit > 150:
            return "📈 رهان قوي - فرصة عالية للربح"
        elif confidence >= 0.5 and profit > 100:
            return "💪 رهان جيد - مخاطرة متوسطة"
        else:
            return "⚠️ رهان تحفظي - مخاطرة عالية"
    
    def generate_betting_report(self, bets: List[Dict]) -> Dict:
        """توليد تقرير الرهان"""
        total_stake = sum(bet.get('stake', 0) for bet in bets)
        total_profit = sum(bet.get('potential_profit', 0) for bet in bets)
        avg_confidence = sum(bet.get('confidence', 0) for bet in bets) / len(bets) if bets else 0
        
        return {
            'total_bets': len(bets),
            'total_stake': total_stake,
            'total_potential_profit': total_profit,
            'roi': (total_profit / total_stake) * 100 if total_stake > 0 else 0,
            'average_confidence': avg_confidence,
            'bet_breakdown': {
                'home_wins': len([b for b in bets if b.get('bet_type') == 'HOME_WIN']),
                'away_wins': len([b for b in bets if b.get('bet_type') == 'AWAY_WIN']),
                'draws': len([b for b in bets if b.get('bet_type') == 'DRAW'])
            },
            'bets': bets
        }

def test_betting_fix():
    """اختبار إصلاح نظام الرهان"""
    fixer = BettingDataFixer()
    betting_engine = SimpleBettingEngine()
    
    # بيانات اختبار
    test_matches = [
        {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea', 
            'match_date': '2025-08-20',
            'week_number': 1,
            'prediction': {'home_goals': 2, 'away_goals': 1, 'confidence': 0.7}
        },
        {
            'home_team': 'Man City',
            'away_team': 'Liverpool',
            'match_date': '2025-08-21', 
            'week_number': 1,
            'prediction': {'home_goals': 1, 'away_goals': 1, 'confidence': 0.6}
        }
    ]
    
    print("🔧 اختبار إصلاح بيانات الرهان...")
    fixed_matches = fixer.fix_match_data_for_betting(test_matches)
    print(f"✅ تم إصلاح {len(fixed_matches)} مباراة")
    
    print("🎰 اختبار محرك الرهان المبسط...")
    bets = []
    for match in fixed_matches:
        bet = betting_engine.calculate_simple_bet(match, stake=100.0)
        bets.append(bet)
        print(f"• {bet['match']}: {bet['prediction']} | رهان: {bet['bet_type']} | ربح محتمل: {bet['potential_profit']}")
    
    report = betting_engine.generate_betting_report(bets)
    print(f"📊 التقرير: {report['total_bets']} رهان | إجمالي ربح محتمل: {report['total_potential_profit']} | ROI: {report['roi']:.1f}%")

if __name__ == "__main__":
    test_betting_fix()