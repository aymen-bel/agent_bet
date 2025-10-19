# advanced_team_assessor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime, timedelta

class AdvancedTeamAssessor:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.teams_assessment = {}
        self.external_factors_cache = {}
        self.motivation_factors = self._initialize_motivation_factors()
        
    def _initialize_motivation_factors(self):
        """تهيئة عوامل التحفيز والدوافع الأساسية"""
        return {
            'relegation_battle': 1.15,      # معركة الهبوط
            'title_race': 1.20,             # سباق اللقب
            'europe_qualification': 1.15,   # التأهل لأوروبا
            'derby_match': 1.25,            # مباراة ديربي
            'revenge_match': 1.10,          # مباراة ثأر
            'new_manager': 1.15,            # مدرب جديد
            'final_stages': 1.10,           # مراحل نهائية
            'mid_season': 1.00,             # منتصف الموسم
            'early_season': 0.95,           # بداية الموسم
        }
    
    def calculate_advanced_metrics(self, team_name, external_context=None):
        """حساب مؤشرات متقدمة مع عوامل خارجية"""
        home_matches = self.data[self.data['HomeTeam'] == team_name]
        away_matches = self.data[self.data['AwayTeam'] == team_name]
        
        if len(home_matches) + len(away_matches) < 10:
            return None
            
        # معالجة العوامل الخارجية
        context = self._process_external_context(team_name, external_context)
        
        return {
            **self._calculate_performance_metrics(home_matches, away_matches, team_name),
            **self._calculate_attacking_metrics(home_matches, away_matches, team_name),
            **self._calculate_defensive_metrics(home_matches, away_matches, team_name),
            **self._calculate_consistency_metrics(home_matches, away_matches, team_name),
            **self._calculate_form_metrics(home_matches, away_matches, team_name),
            **self._calculate_motivation_metrics(team_name, context),
            **self._calculate_external_factors(team_name, context)
        }
    
    def _process_external_context(self, team_name, external_context):
        """معالجة السياق الخارجي للفريق"""
        default_context = {
            'current_position': 10,
            'league_context': 'mid_season',
            'recent_events': [],
            'upcoming_importance': 1.0,
            'rival_teams': [],
            'manager_stability': 1.0,
            'injury_crisis': 0
        }
        
        if external_context:
            default_context.update(external_context)
        
        # تحليل تلقائي للسياق بناءً على البيانات
        default_context.update(self._analyze_automatic_context(team_name))
        
        return default_context
    
    def _analyze_automatic_context(self, team_name):
        """تحليل السياق تلقائياً من البيانات"""
        context = {}
        
        # تحليل مرحلة الموسم
        if not self.data.empty and 'Date' in self.data.columns:
            try:
                latest_date = pd.to_datetime(self.data['Date']).max()
                if latest_date.month in [8, 9]:
                    context['league_context'] = 'early_season'
                elif latest_date.month in [4, 5]:
                    context['league_context'] = 'final_stages'
                else:
                    context['league_context'] = 'mid_season'
            except:
                context['league_context'] = 'mid_season'
        
        # تحليل استقرار المدرب (محاكاة)
        context['manager_stability'] = np.random.uniform(0.8, 1.0)
        
        # تحليل أزمة الإصابات (محاكاة)
        context['injury_crisis'] = np.random.choice([0, 0, 0, 1, 2], p=[0.6, 0.2, 0.1, 0.05, 0.05])
        
        return context
    
    def _calculate_performance_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات الأداء الأساسية المحسنة"""
        total_matches = len(home_matches) + len(away_matches)
        
        # حساب النقاط المعدلة مع تصحيح القوة
        home_points = len(home_matches[home_matches['FTR'] == 'H']) * 3 + len(home_matches[home_matches['FTR'] == 'D'])
        away_points = len(away_matches[away_matches['FTR'] == 'A']) * 3 + len(away_matches[away_matches['FTR'] == 'D'])
        total_points = home_points + away_points
        
        # حساب قوة الخصوم لتعديل النقاط
        opponent_strength_home = self._calculate_opponent_strength(home_matches, 'AwayTeam')
        opponent_strength_away = self._calculate_opponent_strength(away_matches, 'HomeTeam')
        
        # النقاط المعدلة حسب قوة الخصوم
        adjusted_home_points = home_points * (1 + opponent_strength_home * 0.1)
        adjusted_away_points = away_points * (1 + opponent_strength_away * 0.1)
        adjusted_total_points = adjusted_home_points + adjusted_away_points
        
        goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        return {
            'total_matches': total_matches,
            'points': total_points,
            'adjusted_points': adjusted_total_points,
            'points_per_match': total_points / total_matches,
            'adjusted_points_per_match': adjusted_total_points / total_matches,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
            'adjusted_goal_difference': (goals_scored - goals_conceded) * (1 + (opponent_strength_home + opponent_strength_away) * 0.05),
            'win_rate': (len(home_matches[home_matches['FTR'] == 'H']) + len(away_matches[away_matches['FTR'] == 'A'])) / total_matches,
            'home_strength': home_points / (len(home_matches) * 3) if len(home_matches) > 0 else 0,
            'away_strength': away_points / (len(away_matches) * 3) if len(away_matches) > 0 else 0,
            'opponent_strength_home': opponent_strength_home,
            'opponent_strength_away': opponent_strength_away,
        }
    
    def _calculate_opponent_strength(self, matches, opponent_col):
        """حساب قوة الخصوم المتوسط"""
        if len(matches) == 0:
            return 0.5
        
        opponent_strengths = []
        for _, match in matches.iterrows():
            opponent = match[opponent_col]
            opponent_strength = self._get_team_strength(opponent)
            opponent_strengths.append(opponent_strength)
        
        return np.mean(opponent_strengths) if opponent_strengths else 0.5
    
    def _get_team_strength(self, team_name):
        """الحصول على قوة الفريق (مخبأة للأداء)"""
        if team_name in self.teams_assessment:
            return self.teams_assessment[team_name].get('comprehensive_score', 50) / 100
        return 0.5
    
    def _calculate_attacking_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات هجومية متقدمة"""
        total_matches = len(home_matches) + len(away_matches)
        goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        
        # بيانات التصويب المتقدمة
        if all(col in home_matches.columns for col in ['HS', 'HST']):
            home_shots = home_matches['HS'].sum()
            home_shots_target = home_matches['HST'].sum()
            away_shots = away_matches['AS'].sum()
            away_shots_target = away_matches['AST'].sum()
            
            total_shots = home_shots + away_shots
            total_shots_target = home_shots_target + away_shots_target
            
            shot_efficiency = goals_scored / total_shots if total_shots > 0 else 0.1
            conversion_rate = goals_scored / total_shots_target if total_shots_target > 0 else 0.15
            shot_accuracy = total_shots_target / total_shots if total_shots > 0 else 0.35
            
            # كفاءة الهجوم في ظروف مختلفة
            attacking_consistency = self._calculate_attacking_consistency(home_matches, away_matches)
        else:
            # قيم افتراضية أكثر واقعية
            shot_efficiency = 0.12
            conversion_rate = 0.15
            shot_accuracy = 0.35
            attacking_consistency = 0.7
        
        return {
            'shot_efficiency': shot_efficiency,
            'conversion_rate': conversion_rate,
            'shot_accuracy': shot_accuracy,
            'goals_per_match': goals_scored / total_matches,
            'attacking_pressure': (home_matches['HS'].mean() if 'HS' in home_matches.columns else 12 + 
                                 away_matches['AS'].mean() if 'AS' in away_matches.columns else 10) / 2,
            'expected_goals': self._calculate_expected_goals(home_matches, away_matches),
            'attacking_consistency': attacking_consistency,
            'big_chances_created': self._estimate_big_chances(home_matches, away_matches)
        }
    
    def _calculate_attacking_consistency(self, home_matches, away_matches):
        """حساب اتساق الهجوم عبر المباريات"""
        home_goals = home_matches['FTHG'].tolist()
        away_goals = away_matches['FTAG'].tolist()
        all_goals = home_goals + away_goals
        
        if len(all_goals) < 2:
            return 0.5
        
        # اتساق الهجوم = 1 - معامل الاختلاف (مع تعديل)
        mean_goals = np.mean(all_goals)
        if mean_goals == 0:
            return 0.3
        
        cv = np.std(all_goals) / mean_goals
        consistency = 1 - min(cv, 1.0)  # الحد الأقصى لمعامل الاختلاف = 1
        
        return max(0.1, consistency)
    
    def _estimate_big_chances(self, home_matches, away_matches):
        """تقدير الفرص الكبيرة (معدل)"""
        total_matches = len(home_matches) + len(away_matches)
        if total_matches == 0:
            return 1.5
        
        # تقدير بناءً على التصويب على المرمى والأهداف
        home_big_chances = home_matches['HST'].mean() if 'HST' in home_matches.columns else 5.0
        away_big_chances = away_matches['AST'].mean() if 'AST' in away_matches.columns else 4.0
        
        return (home_big_chances + away_big_chances) / 2
    
    def _calculate_defensive_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات دفاعية متقدمة"""
        total_matches = len(home_matches) + len(away_matches)
        goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        if all(col in home_matches.columns for col in ['AS', 'AST']):
            home_shots_faced = home_matches['AS'].sum()
            home_shots_target_faced = home_matches['AST'].sum()
            away_shots_faced = away_matches['HS'].sum()
            away_shots_target_faced = away_matches['HST'].sum()
            
            total_shots_faced = home_shots_faced + away_shots_faced
            total_shots_target_faced = home_shots_target_faced + away_shots_target_faced
            
            defensive_efficiency = 1 - (goals_conceded / total_shots_target_faced) if total_shots_target_faced > 0 else 0.7
            pressure_resistance = 1 - (total_shots_target_faced / total_shots_faced) if total_shots_faced > 0 else 0.3
        else:
            defensive_efficiency = 0.7
            pressure_resistance = 0.3
        
        clean_sheets = len(home_matches[home_matches['FTAG'] == 0]) + len(away_matches[away_matches['FTHG'] == 0])
        
        return {
            'defensive_efficiency': defensive_efficiency,
            'pressure_resistance': pressure_resistance,
            'clean_sheet_rate': clean_sheets / total_matches,
            'goals_conceded_per_match': goals_conceded / total_matches,
            'defensive_stability': 1 - (goals_conceded / total_matches) / 3,
            'defensive_consistency': self._calculate_defensive_consistency(home_matches, away_matches)
        }
    
    def _calculate_defensive_consistency(self, home_matches, away_matches):
        """حساب اتساق الدفاع"""
        home_conceded = home_matches['FTAG'].tolist()
        away_conceded = away_matches['FTHG'].tolist()
        all_conceded = home_conceded + away_conceded
        
        if len(all_conceded) < 2:
            return 0.5
        
        # الدفاع المتسق يكون ذو تباين منخفض في الأهداف المسجلة عليه
        consistency = 1 - (np.std(all_conceded) / max(1, np.mean(all_conceded)))
        return max(0.1, min(1.0, consistency))
    
    def _calculate_expected_goals(self, home_matches, away_matches):
        """حساب xG مبسط"""
        home_xg = (home_matches['HST'].sum() * 0.3 if 'HST' in home_matches.columns else len(home_matches) * 1.2)
        away_xg = (away_matches['AST'].sum() * 0.25 if 'AST' in away_matches.columns else len(away_matches) * 1.0)
        return home_xg + away_xg
    
    def _calculate_consistency_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات الاتساق المتقدمة"""
        # اتساق النتائج
        home_results = [1 if res == 'H' else 0.5 if res == 'D' else 0 for res in home_matches['FTR']]
        away_results = [1 if res == 'A' else 0.5 if res == 'D' else 0 for res in away_matches['FTR']]
        all_results = home_results + away_results
        
        if len(all_results) < 2:
            return {'consistency_score': 0.5, 'volatility': 0.5, 'performance_trend': 0}
        
        consistency = 1 - np.std(all_results)
        
        # حساب اتجاه الأداء
        performance_trend = self._calculate_performance_trend(home_matches, away_matches, team_name)
        
        return {
            'consistency_score': max(0, min(1, consistency)),
            'volatility': np.std(all_results),
            'performance_trend': performance_trend
        }
    
    def _calculate_performance_trend(self, home_matches, away_matches, team_name):
        """حساب اتجاه الأداء (تحسين مستمر/تراجع)"""
        # تقسيم المباريات إلى ثلاث فترات
        total_matches = len(home_matches) + len(away_matches)
        if total_matches < 6:
            return 0
        
        # ترتيب المباريات حسب التاريخ (افتراضي)
        all_matches = pd.concat([home_matches, away_matches])
        if 'Date' in all_matches.columns:
            all_matches = all_matches.sort_values('Date')
        
        # تقسيم إلى ثلاث فترات متساوية
        period_size = max(1, total_matches // 3)
        periods = []
        
        for i in range(3):
            start_idx = i * period_size
            end_idx = min((i + 1) * period_size, total_matches)
            period_matches = all_matches.iloc[start_idx:end_idx]
            periods.append(period_matches)
        
        # حساب النقاط في كل فترة
        period_points = []
        for period in periods:
            points = 0
            for _, match in period.iterrows():
                if match['HomeTeam'] == team_name:
                    if match['FTR'] == 'H': points += 3
                    elif match['FTR'] == 'D': points += 1
                else:
                    if match['FTR'] == 'A': points += 3
                    elif match['FTR'] == 'D': points += 1
            period_points.append(points / len(period) if len(period) > 0 else 0)
        
        # حساب الاتجاه (انحدار خطي بسيط)
        if len(period_points) >= 2:
            x = np.arange(len(period_points))
            trend = np.polyfit(x, period_points, 1)[0]  # الميل
            return trend * 2  # تضخيم للتأثير
        else:
            return 0
    
    def _calculate_form_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات الشكل الحالي المحسنة"""
        # آخر 8 مباريات مرتبة حسب التاريخ
        recent_home = home_matches.tail(4)
        recent_away = away_matches.tail(4)
        recent_matches = pd.concat([recent_home, recent_away])
        
        if 'Date' in recent_matches.columns:
            recent_matches = recent_matches.sort_values('Date').tail(8)
        else:
            recent_matches = recent_matches.tail(8)
        
        if len(recent_matches) == 0:
            return {'current_form': 0.5, 'form_trend': 0, 'form_momentum': 0}
        
        # حساب النقاط في المباريات الحديثة
        recent_points = 0
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team_name:
                if match['FTR'] == 'H': recent_points += 3
                elif match['FTR'] == 'D': recent_points += 1
            else:
                if match['FTR'] == 'A': recent_points += 3
                elif match['FTR'] == 'D': recent_points += 1
        
        current_form = recent_points / (len(recent_matches) * 3)
        
        # حساب الاتجاه مع وزن أعلى للمباريات الأحدث
        if len(recent_matches) >= 4:
            weights = [0.1, 0.15, 0.25, 0.5]  # أوزان تصاعدية للأحدث
            weighted_points = 0
            total_weight = 0
            
            for i, (_, match) in enumerate(recent_matches.tail(4).iterrows()):
                weight = weights[i]
                if match['HomeTeam'] == team_name:
                    if match['FTR'] == 'H': weighted_points += 3 * weight
                    elif match['FTR'] == 'D': weighted_points += 1 * weight
                else:
                    if match['FTR'] == 'A': weighted_points += 3 * weight
                    elif match['FTR'] == 'D': weighted_points += 1 * weight
                total_weight += weight
            
            form_momentum = weighted_points / (total_weight * 3) if total_weight > 0 else 0.5
        else:
            form_momentum = current_form
        
        return {
            'current_form': current_form,
            'form_trend': self._calculate_performance_trend(home_matches, away_matches, team_name),
            'form_momentum': form_momentum
        }
    
    def _calculate_motivation_metrics(self, team_name, context):
        """حساب مؤشرات التحفيز والدوافع"""
        motivation_score = 1.0
        
        # تطبيق عوامل التحفيز بناءً على السياق
        league_context = context.get('league_context', 'mid_season')
        motivation_score *= self.motivation_factors.get(league_context, 1.0)
        
        # تأثير استقرار المدرب
        manager_stability = context.get('manager_stability', 1.0)
        if manager_stability < 0.7:
            motivation_score *= 1.15  # تأثير المدرب الجديد
        elif manager_stability > 0.9:
            motivation_score *= 1.05  # استقرار إيجابي
        
        # تأثير مرحلة الموسم على الدوافع
        seasonal_motivation = self._calculate_seasonal_motivation(context)
        motivation_score *= seasonal_motivation
        
        return {
            'motivation_factor': motivation_score,
            'manager_stability': manager_stability,
            'seasonal_motivation': seasonal_motivation,
            'pressure_handling': self._estimate_pressure_handling(team_name, context)
        }
    
    def _calculate_seasonal_motivation(self, context):
        """حساب الدوافع الموسمية"""
        league_context = context.get('league_context', 'mid_season')
        current_position = context.get('current_position', 10)
        
        if league_context == 'final_stages':
            if current_position <= 4:
                return 1.20  # سباق اللقب والتأهل للأبطال
            elif current_position <= 6:
                return 1.15  # التأهل لأوروبا
            elif current_position >= 18:
                return 1.25  # معركة الهبوط
            else:
                return 1.05  # منتصف الجدول
        elif league_context == 'early_season':
            return 0.95  # بداية الموسم
        else:
            return 1.00  # منتصف الموسم
    
    def _estimate_pressure_handling(self, team_name, context):
        """تقدير قدرة الفريق على تحمل الضغط"""
        # محاكاة بناءً على أداء الفريق في المباريات المهمة
        pressure_performance = np.random.uniform(0.6, 1.0)
        
        # تعديل بناءً على الخبرة (افتراضية)
        if any(exp in team_name.lower() for exp in ['city', 'united', 'liverpool', 'arsenal', 'chelsea']):
            pressure_performance *= 1.1  # فرق كبيرة أكثر خبرة
        
        return min(1.0, pressure_performance)
    
    def _calculate_external_factors(self, team_name, context):
        """حساب العوامل الخارجية المؤثرة"""
        injury_impact = self._calculate_injury_impact(context.get('injury_crisis', 0))
        fixture_congestion = self._calculate_fixture_congestion(team_name)
        travel_fatigue = self._estimate_travel_fatigue(team_name)
        
        external_factor = injury_impact * fixture_congestion * travel_fatigue
        
        return {
            'injury_impact': injury_impact,
            'fixture_congestion': fixture_congestion,
            'travel_fatigue': travel_fatigue,
            'external_factor': external_factor,
            'overall_context_impact': external_factor * context.get('manager_stability', 1.0)
        }
    
    def _calculate_injury_impact(self, injury_crisis_level):
        """حساب تأثير الإصابات"""
        # مستوى 0: لا إصابات مهمة (تأثير 1.0)
        # مستوى 1: إصابات طفيفة (تأثير 0.95)
        # مستوى 2: إصابات متوسطة (تأثير 0.85)
        # مستوى 3+: أزمة إصابات (تأثير 0.7)
        if injury_crisis_level == 0:
            return 1.0
        elif injury_crisis_level == 1:
            return 0.95
        elif injury_crisis_level == 2:
            return 0.85
        else:
            return 0.70
    
    def _calculate_fixture_congestion(self, team_name):
        """تقدير ازدحام المباريات (محاكاة)"""
        # محاكاة ازدحام المباريات بناءً على اسم الفريق
        base_congestion = np.random.uniform(0.9, 1.0)
        
        # فرق كبيرة عادةً ما يكون لديها ازدحام أكثر
        if any(big_team in team_name.lower() for big_team in ['city', 'united', 'liverpool', 'chelsea', 'arsenal']):
            base_congestion *= 0.92  # تأثير أكبر للازدحام
        
        return base_congestion
    
    def _estimate_travel_fatigue(self, team_name):
        """تقدير تأثير السفر (محاكاة)"""
        # محاكاة تأثير السفر بناءً على موقع الفريق الافتراضي
        travel_impact = np.random.uniform(0.95, 1.0)
        return travel_impact
    
    def calculate_points_from_matches(self, matches, team_name):
        """حساب النقاط من مجموعة مباريات"""
        points = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team_name:
                if match['FTR'] == 'H': points += 3
                elif match['FTR'] == 'D': points += 1
            else:
                if match['FTR'] == 'A': points += 3
                elif match['FTR'] == 'D': points += 1
        return points


# enhanced_team_assessor.py
import pandas as pd
import numpy as np

class EnhancedTeamAssessor:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.teams_assessment = {}
        self.advanced_assessor = AdvancedTeamAssessor(data_file)
        
    def calculate_realistic_metrics(self, team_name, external_context=None):
        """حساب مؤشرات واقعية مع عوامل خارجية"""
        home_matches = self.data[self.data['HomeTeam'] == team_name]
        away_matches = self.data[self.data['AwayTeam'] == team_name]
        
        total_matches = len(home_matches) + len(away_matches)
        if total_matches < 5:
            return None
        
        # الحصول على التحليل المتقدم
        advanced_metrics = self.advanced_assessor.calculate_advanced_metrics(team_name, external_context)
        
        # النتائج الأساسية الواقعية
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        draws = len(home_matches[home_matches['FTR'] == 'D']) + len(away_matches[away_matches['FTR'] == 'D'])
        
        total_points = (home_wins + away_wins) * 3 + draws
        goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        # مؤشرات واقعية (بدون تضخيم)
        base_metrics = {
            'total_matches': total_matches,
            'points': total_points,
            'points_per_match': total_points / total_matches,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
            'win_rate': (home_wins + away_wins) / total_matches,
            'draw_rate': draws / total_matches,
            'loss_rate': (total_matches - home_wins - away_wins - draws) / total_matches,
            'goals_per_match': goals_scored / total_matches,
            'conceded_per_match': goals_conceded / total_matches,
        }
        
        # مؤشرات الهجوم الواقعية
        attack_metrics = self._calculate_realistic_attack(home_matches, away_matches, total_matches, goals_scored)
        
        # مؤشرات الدفاع الواقعية
        defense_metrics = self._calculate_realistic_defense(home_matches, away_matches, total_matches, goals_conceded)
        
        # مؤشرات الأداء
        performance_metrics = self._calculate_performance_metrics(home_matches, away_matches, team_name)
        
        # دمج مع المقاييس المتقدمة
        all_metrics = {**base_metrics, **attack_metrics, **defense_metrics, **performance_metrics}
        
        if advanced_metrics:
            all_metrics.update({
                'adjusted_points_per_match': advanced_metrics.get('adjusted_points_per_match', all_metrics['points_per_match']),
                'motivation_factor': advanced_metrics.get('motivation_factor', 1.0),
                'external_factor': advanced_metrics.get('external_factor', 1.0),
                'current_form': advanced_metrics.get('current_form', 0.5),
                'form_momentum': advanced_metrics.get('form_momentum', 0.5),
                'performance_trend': advanced_metrics.get('performance_trend', 0),
                'consistency_score': advanced_metrics.get('consistency_score', 0.5)
            })
        
        return all_metrics
    
    def _calculate_realistic_attack(self, home_matches, away_matches, total_matches, goals_scored):
        """مؤشرات هجوم واقعية"""
        # استخدام بيانات التصويب إذا كانت متاحة، وإلا استخدام قيم واقعية
        if all(col in home_matches.columns for col in ['HS', 'HST']):
            home_shots = home_matches['HS'].sum()
            home_shots_target = home_matches['HST'].sum()
            away_shots = away_matches['AS'].sum()
            away_shots_target = away_matches['AST'].sum()
            
            total_shots = home_shots + away_shots
            total_shots_target = home_shots_target + away_shots_target
            
            shot_efficiency = goals_scored / total_shots if total_shots > 0 else 0.08
            conversion_rate = goals_scored / total_shots_target if total_shots_target > 0 else 0.12
            shot_accuracy = total_shots_target / total_shots if total_shots > 0 else 0.35
        else:
            # قيم واقعية بناءً على إحصائيات الدوري الإنجليزي
            shot_efficiency = 0.10  # 10% كفاءة تصويب
            conversion_rate = 0.15  # 15% تحويل
            shot_accuracy = 0.35    # 35% دقة
        
        return {
            'shot_efficiency': shot_efficiency,
            'conversion_rate': conversion_rate,
            'shot_accuracy': shot_accuracy,
            'attacking_pressure': (home_matches['HS'].mean() if 'HS' in home_matches.columns else 12 + 
                                 away_matches['AS'].mean() if 'AS' in away_matches.columns else 9) / 20
        }
    
    def _calculate_realistic_defense(self, home_matches, away_matches, total_matches, goals_conceded):
        """مؤشرات دفاع واقعية"""
        clean_sheets = len(home_matches[home_matches['FTAG'] == 0]) + len(away_matches[away_matches['FTHG'] == 0])
        
        if all(col in home_matches.columns for col in ['AS', 'AST']):
            home_shots_faced = home_matches['AS'].sum()
            home_shots_target_faced = home_matches['AST'].sum()
            away_shots_faced = away_matches['HS'].sum()
            away_shots_target_faced = away_matches['HST'].sum()
            
            total_shots_target_faced = home_shots_target_faced + away_shots_target_faced
            
            defensive_efficiency = 1 - (goals_conceded / total_shots_target_faced) if total_shots_target_faced > 0 else 0.65
        else:
            defensive_efficiency = 0.65  # 65% كفاءة دفاعية
        
        return {
            'defensive_efficiency': defensive_efficiency,
            'clean_sheet_rate': clean_sheets / total_matches,
            'goals_conceded_per_match': goals_conceded / total_matches
        }
    
    def _calculate_performance_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات أداء واقعية"""
        home_points = len(home_matches[home_matches['FTR'] == 'H']) * 3 + len(home_matches[home_matches['FTR'] == 'D'])
        away_points = len(away_matches[away_matches['FTR'] == 'A']) * 3 + len(away_matches[away_matches['FTR'] == 'D'])
        
        home_strength = home_points / (len(home_matches) * 3) if len(home_matches) > 0 else 0.4
        away_strength = away_points / (len(away_matches) * 3) if len(away_matches) > 0 else 0.3
        
        # حساب الشكل الحالي (آخر 8 مباريات)
        recent_matches = pd.concat([home_matches.tail(4), away_matches.tail(4)]).tail(8)
        recent_points = 0
        
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team_name:
                if match['FTR'] == 'H': recent_points += 3
                elif match['FTR'] == 'D': recent_points += 1
            else:
                if match['FTR'] == 'A': recent_points += 3
                elif match['FTR'] == 'D': recent_points += 1
        
        current_form = recent_points / (len(recent_matches) * 3) if len(recent_matches) > 0 else 0.5
        
        # حساب الاتساق في الأداء
        consistency = self._calculate_consistency(home_matches, away_matches)
        
        return {
            'home_advantage': home_strength,
            'away_resilience': away_strength,
            'current_form': current_form,
            'performance_balance': (home_strength + away_strength) / 2,
            'consistency': consistency
        }
    
    def _calculate_consistency(self, home_matches, away_matches):
        """حساب اتساق الأداء"""
        # نتائج المباريات كنقاط
        home_results = []
        for _, match in home_matches.iterrows():
            if match['FTR'] == 'H': home_results.append(3)
            elif match['FTR'] == 'D': home_results.append(1)
            else: home_results.append(0)
        
        away_results = []
        for _, match in away_matches.iterrows():
            if match['FTR'] == 'A': away_results.append(3)
            elif match['FTR'] == 'D': away_results.append(1)
            else: away_results.append(0)
        
        all_results = home_results + away_results
        
        if len(all_results) < 2:
            return 0.5
        
        # الاتساق = 1 - معامل الاختلاف
        consistency = 1 - (np.std(all_results) / np.mean(all_results)) if np.mean(all_results) > 0 else 0.5
        return max(0, min(1, consistency))
    
    def calculate_realistic_score(self, metrics):
        """حساب درجة واقعية محسنة (0-100) مع العوامل الخارجية"""
        if not metrics:
            return 0
        
        # أوزان محسنة تشمل العوامل الخارجية
        weights = {
            'points_per_match': 0.20,          # 20%
            'adjusted_points_per_match': 0.15, # 15%
            'win_rate': 0.15,                  # 15%
            'goal_difference': 0.10,           # 10%
            'defensive_efficiency': 0.10,      # 10%
            'current_form': 0.08,              # 8%
            'form_momentum': 0.07,             # 7%
            'motivation_factor': 0.05,         # 5%
            'consistency_score': 0.05,         # 5%
            'performance_trend': 0.05          # 5%
        }
        
        score = 0
        
        # تطبيق الأوزان مع قيم واقعية
        base_ppm = metrics.get('points_per_match', 0)
        adj_ppm = metrics.get('adjusted_points_per_match', base_ppm)
        
        score += min(base_ppm * 7, 20)                    # 3 نقاط لكل مباراة = 21
        score += min(adj_ppm * 6, 15)                    # نقاط معدلة
        score += metrics.get('win_rate', 0) * 15          # 100% فوز = 15
        score += min((metrics.get('goal_difference', 0) / max(1, metrics['total_matches'])) * 4, 10)
        score += metrics.get('defensive_efficiency', 0) * 10
        score += metrics.get('current_form', 0.5) * 8
        score += metrics.get('form_momentum', 0.5) * 7
        score += (metrics.get('motivation_factor', 1.0) - 1) * 20 + 5  # تحويل 1.0-1.2 إلى 5-9
        score += metrics.get('consistency_score', 0.5) * 5
        score += min(max(metrics.get('performance_trend', 0) * 10 + 5, 0), 5)  # اتجاه -0.5 إلى +0.5 → 0 إلى 10
        
        # تطبيق العامل الخارجي
        external_factor = metrics.get('external_factor', 1.0)
        score *= external_factor
        
        return min(100, max(0, score))
    
    def assess_all_teams(self, external_contexts=None):
        """تقييم جميع الفرق مع السياقات الخارجية"""
        print("🎯 جاري تقييم جميع الفرق (مع العوامل الخارجية)...")
        
        # تنظيف أسماء الفرق
        self.data['HomeTeam'] = self.data['HomeTeam'].astype(str).str.strip()
        self.data['AwayTeam'] = self.data['AwayTeam'].astype(str).str.strip()
        
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        
        for team in all_teams:
            if pd.isna(team) or team == 'nan':
                continue
            
            # الحصول على السياق الخارجي للفريق إذا كان متوفراً
            team_context = None
            if external_contexts and team in external_contexts:
                team_context = external_contexts[team]
                
            metrics = self.calculate_realistic_metrics(team, team_context)
            
            if metrics:
                comprehensive_score = self.calculate_realistic_score(metrics)
                metrics['comprehensive_score'] = comprehensive_score
                self.teams_assessment[team] = metrics
        
        print(f"✅ تم تقييم {len(self.teams_assessment)} فريق مع العوامل الخارجية")
        return self.teams_assessment
    
    def create_final_ranking(self):
        """إنشاء ترتيب نهائي واقعي مع العوامل الخارجية"""
        if not self.teams_assessment:
            self.assess_all_teams()
        
        ranking_data = []
        for team, metrics in self.teams_assessment.items():
            row_data = {'Team': team}
            row_data.update(metrics)
            ranking_data.append(row_data)
        
        final_ranking = pd.DataFrame(ranking_data)
        final_ranking = final_ranking.sort_values('comprehensive_score', ascending=False)
        
        return final_ranking
    
    def generate_detailed_report(self):
        """إنشاء تقرير مفصل واقعي مع العوامل الخارجية"""
        if not self.teams_assessment:
            self.assess_all_teams()
        
        print("\n" + "="*80)
        print("📊 التقرير الشامل لتقييم الفرق (واقعي مع العوامل الخارجية)")
        print("="*80)
        
        # أفضل 10 فرق
        top_10 = sorted(self.teams_assessment.items(), key=lambda x: x[1]['comprehensive_score'], reverse=True)[:10]
        
        print(f"\n🏆 أفضل 10 فرق بناءً على التقييم الشامل:")
        print("-" * 80)
        for i, (team, metrics) in enumerate(top_10, 1):
            motivation = metrics.get('motivation_factor', 1.0)
            external = metrics.get('external_factor', 1.0)
            print(f"{i:2d}. {team:<20} | {metrics['comprehensive_score']:5.1f} نقطة | "
                  f"نقاط: {metrics['points_per_match']:4.2f} | فوز: {metrics['win_rate']:5.1%} | "
                  f"تحفيز: {motivation:4.2f} | خارجي: {external:4.2f}")
        
        # إحصائيات عامة
        print(f"\n📈 الإحصائيات العامة (مع العوامل الخارجية):")
        scores = [metrics['comprehensive_score'] for metrics in self.teams_assessment.values()]
        points = [metrics['points_per_match'] for metrics in self.teams_assessment.values()]
        motivations = [metrics.get('motivation_factor', 1.0) for metrics in self.teams_assessment.values()]
        externals = [metrics.get('external_factor', 1.0) for metrics in self.teams_assessment.values()]
        
        print(f"• عدد الفرق المُقيمة: {len(self.teams_assessment)}")
        print(f"• متوسط الدرجة الشاملة: {np.mean(scores):.1f}")
        print(f"• أعلى درجة: {np.max(scores):.1f}")
        print(f"• أدنى درجة: {np.min(scores):.1f}")
        print(f"• متوسط النقاط: {np.mean(points):.2f} لكل مباراة")
        print(f"• متوسط عامل التحفيز: {np.mean(motivations):.2f}")
        print(f"• متوسط العامل الخارجي: {np.mean(externals):.2f}")
        
        # الفرق المتميزة
        print(f"\n🎯 الفرق المتميزة في مجالات محددة:")
        
        # أفضل هجوم
        best_attack = max(self.teams_assessment.items(), 
                         key=lambda x: x[1].get('shot_efficiency', 0))
        # أفضل دفاع
        best_defense = max(self.teams_assessment.items(), 
                          key=lambda x: x[1].get('defensive_efficiency', 0))
        # أفضل تحفيز
        best_motivation = max(self.teams_assessment.items(), 
                             key=lambda x: x[1].get('motivation_factor', 1.0))
        
        print(f"• أفضل هجوم: {best_attack[0]} ({best_attack[1]['shot_efficiency']:.1%})")
        print(f"• أفضل دفاع: {best_defense[0]} ({best_defense[1]['defensive_efficiency']:.1%})")
        print(f"• أعلى تحفيز: {best_motivation[0]} ({best_motivation[1]['motivation_factor']:.2f})")
        
        # الفرق الأكثر اتساقاً
        most_consistent = sorted(self.teams_assessment.items(), 
                               key=lambda x: x[1].get('consistency_score', x[1].get('consistency', 0)), 
                               reverse=True)[:3]
        print(f"\n⚡ الفرق الأكثر اتساقاً:")
        for team, metrics in most_consistent:
            consistency = metrics.get('consistency_score', metrics.get('consistency', 0))
            print(f"• {team}: اتساق {consistency:.1%}")

# مثال على الاستخدام
if __name__ == "__main__":
    # اختبار الكلاس
    assessor = EnhancedTeamAssessor("data/football-data/combined_seasons_data.csv")
    
    # تعريف سياقات خارجية مثال
    external_contexts = {
        "Man City": {'current_position': 1, 'league_context': 'final_stages', 'manager_stability': 0.9},
        "Liverpool": {'current_position': 2, 'league_context': 'final_stages', 'manager_stability': 0.8},
        "Leicester": {'current_position': 18, 'league_context': 'final_stages', 'manager_stability': 0.7, 'injury_crisis': 2}
    }
    
    # تقييم جميع الفرق مع السياقات الخارجية
    assessment = assessor.assess_all_teams(external_contexts)
    
    # إنشاء الترتيب
    ranking = assessor.create_final_ranking()
    
    # عرض التقرير
    assessor.generate_detailed_report()
    
    print(f"\n✅ اكتمل التقييم بنجاح مع تحليل العوامل الخارجية!")