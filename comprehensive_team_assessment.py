# comprehensive_team_assessment.py (محسّن)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveTeamAssessment:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.teams_assessment = {}
        self.final_ranking = None
        
    def preprocess_data(self):
        """معالجة متقدمة للبيانات"""
        print("🔧 جاري المعالجة المتقدمة للبيانات...")
        
        # تنظيف البيانات
        self.data['HomeTeam'] = self.data['HomeTeam'].astype(str).str.strip()
        self.data['AwayTeam'] = self.data['AwayTeam'].astype(str).str.strip()
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        
        # إزالة المباريات غير المكتملة
        required_columns = ['FTHG', 'FTAG', 'FTR', 'HomeTeam', 'AwayTeam']
        self.data = self.data.dropna(subset=required_columns)
        
        # ملء القيم المفقودة
        stat_columns = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
        for col in stat_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        print(f"✅ تمت معالجة {len(self.data)} مباراة")
        return self.data
    
    def calculate_enhanced_metrics(self, team_name):
        """حساب مؤشرات محسنة ومتكاملة"""
        home_matches = self.data[self.data['HomeTeam'] == team_name]
        away_matches = self.data[self.data['AwayTeam'] == team_name]
        
        total_matches = len(home_matches) + len(away_matches)
        if total_matches < 5:
            return None
            
        # 🎯 المؤشرات الأساسية المحسنة
        basic_metrics = self._calculate_enhanced_basic_metrics(home_matches, away_matches, team_name)
        
        # ⚽ مؤشرات الهجوم المتقدمة
        attack_metrics = self._calculate_enhanced_attack_metrics(home_matches, away_matches, team_name)
        
        # 🛡️ مؤشرات الدفاع المتقدمة  
        defense_metrics = self._calculate_enhanced_defense_metrics(home_matches, away_matches, team_name)
        
        # 📊 مؤشرات الأداء التكتيكي
        tactical_metrics = self._calculate_tactical_metrics(home_matches, away_matches, team_name)
        
        # 🔄 مؤشرات الاتساق والاستقرار
        consistency_metrics = self._calculate_enhanced_consistency_metrics(home_matches, away_matches, team_name)
        
        # 🏟️ مؤشرات الظروف (منزل/خارج)
        situational_metrics = self._calculate_situational_metrics(home_matches, away_matches, team_name)
        
        return {
            **basic_metrics,
            **attack_metrics,
            **defense_metrics,
            **tactical_metrics,
            **consistency_metrics,
            **situational_metrics
        }
    
    def _calculate_enhanced_basic_metrics(self, home_matches, away_matches, team_name):
        """المؤشرات الأساسية المحسنة"""
        total_matches = len(home_matches) + len(away_matches)
        
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        draws = len(home_matches[home_matches['FTR'] == 'D']) + len(away_matches[away_matches['FTR'] == 'D'])
        
        total_points = (home_wins + away_wins) * 3 + draws
        goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        return {
            'total_matches': total_matches,
            'points': total_points,
            'wins': home_wins + away_wins,
            'draws': draws,
            'losses': total_matches - (home_wins + away_wins + draws),
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
            'points_per_match': total_points / total_matches,
            'win_rate': (home_wins + away_wins) / total_matches,
            'draw_rate': draws / total_matches,
            'loss_rate': (total_matches - (home_wins + away_wins + draws)) / total_matches,
            'goals_per_match': goals_scored / total_matches,
            'conceded_per_match': goals_conceded / total_matches
        }
    
    def _calculate_enhanced_attack_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات هجومية محسنة"""
        # بيانات التصويب
        home_shots = home_matches['HS'].sum() if 'HS' in home_matches.columns else len(home_matches) * 12
        away_shots = away_matches['AS'].sum() if 'AS' in away_matches.columns else len(away_matches) * 10
        total_shots = home_shots + away_shots
        
        home_shots_target = home_matches['HST'].sum() if 'HST' in home_matches.columns else home_shots * 0.35
        away_shots_target = away_matches['AST'].sum() if 'AST' in away_matches.columns else away_shots * 0.35
        total_shots_target = home_shots_target + away_shots_target
        
        home_goals = home_matches['FTHG'].sum()
        away_goals = away_matches['FTAG'].sum()
        total_goals = home_goals + away_goals
        
        # بيانات الركنيات والتمريرات
        home_corners = home_matches['HC'].sum() if 'HC' in home_matches.columns else len(home_matches) * 5
        away_corners = away_matches['AC'].sum() if 'AC' in away_matches.columns else len(away_matches) * 4
        total_corners = home_corners + away_corners
        
        return {
            # كفاءة الهجوم
            'shot_efficiency': total_goals / total_shots if total_shots > 0 else 0,
            'shot_accuracy': total_shots_target / total_shots if total_shots > 0 else 0,
            'conversion_rate': total_goals / total_shots_target if total_shots_target > 0 else 0,
            'goals_per_shot': total_goals / total_shots if total_shots > 0 else 0,
            
            # ضغط الهجوم
            'attacking_pressure': total_shots / (len(home_matches) + len(away_matches)),
            'corners_per_match': total_corners / (len(home_matches) + len(away_matches)),
            'shot_volume': total_shots / (len(home_matches) + len(away_matches)),
            
            # توقع الأهداف
            'expected_goals_ratio': self._calculate_advanced_xg_ratio(home_matches, away_matches, total_goals),
            'overperformance_ratio': self._calculate_overperformance_ratio(home_matches, away_matches, total_goals)
        }
    
    def _calculate_enhanced_defense_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات دفاعية محسنة"""
        home_goals_conceded = home_matches['FTAG'].sum()
        away_goals_conceded = away_matches['FTHG'].sum()
        total_goals_conceded = home_goals_conceded + away_goals_conceded
        
        home_shots_faced = home_matches['AS'].sum() if 'AS' in home_matches.columns else len(home_matches) * 10
        away_shots_faced = away_matches['HS'].sum() if 'HS' in away_matches.columns else len(away_matches) * 12
        total_shots_faced = home_shots_faced + away_shots_faced
        
        home_shots_target_faced = home_matches['AST'].sum() if 'AST' in home_matches.columns else home_shots_faced * 0.35
        away_shots_target_faced = away_matches['HST'].sum() if 'HST' in away_matches.columns else away_shots_faced * 0.35
        total_shots_target_faced = home_shots_target_faced + away_shots_target_faced
        
        # الأخطاء والبطاقات
        home_fouls = home_matches['HF'].sum() if 'HF' in home_matches.columns else len(home_matches) * 12
        away_fouls = away_matches['AF'].sum() if 'AF' in away_matches.columns else len(away_matches) * 11
        total_fouls = home_fouls + away_fouls
        
        home_yellow = home_matches['HY'].sum() if 'HY' in home_matches.columns else len(home_matches) * 1.5
        away_yellow = away_matches['AY'].sum() if 'AY' in away_matches.columns else len(away_matches) * 1.8
        total_yellow = home_yellow + away_yellow
        
        return {
            # كفاءة الدفاع
            'defensive_efficiency': 1 - (total_goals_conceded / total_shots_target_faced) if total_shots_target_faced > 0 else 0,
            'goals_conceded_per_shot': total_goals_conceded / total_shots_faced if total_shots_faced > 0 else 0,
            'clean_sheet_rate': self._calculate_clean_sheet_rate(home_matches, away_matches),
            'defensive_stability': 1 - (total_goals_conceded / (len(home_matches) + len(away_matches))) / 3,
            
            # ضغط الدفاع
            'pressure_resistance': total_shots_target_faced / (len(home_matches) + len(away_matches)),
            'fouls_per_match': total_fouls / (len(home_matches) + len(away_matches)),
            'disciplinary_index': total_yellow / (len(home_matches) + len(away_matches))
        }
    
    def _calculate_tactical_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات تكتيكية"""
        total_matches = len(home_matches) + len(away_matches)
        
        # السيطرة على المباراة
        home_possession = home_matches['HC'].mean() if 'HC' in home_matches.columns else 50
        away_possession = away_matches['AC'].mean() if 'AC' in away_matches.columns else 45
        avg_possession = (home_possession * len(home_matches) + away_possession * len(away_matches)) / total_matches
        
        # التمركز الهجومي vs الدفاعي
        home_attack_ratio = home_matches['HST'].mean() / home_matches['HS'].mean() if 'HS' in home_matches.columns and home_matches['HS'].mean() > 0 else 0.35
        away_attack_ratio = away_matches['AST'].mean() / away_matches['AS'].mean() if 'AS' in away_matches.columns and away_matches['AS'].mean() > 0 else 0.35
        
        return {
            'possession_control': avg_possession / 100,
            'attack_efficiency_ratio': (home_attack_ratio + away_attack_ratio) / 2,
            'tactical_discipline': 1 - (self._calculate_tactical_errors(home_matches, away_matches) / total_matches),
            'game_management': self._calculate_game_management(home_matches, away_matches)
        }
    
    def _calculate_enhanced_consistency_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات اتساق محسنة"""
        # اتساق النتائج
        home_results = [3 if result == 'H' else 1 if result == 'D' else 0 for result in home_matches['FTR']]
        away_results = [3 if result == 'A' else 1 if result == 'D' else 0 for result in away_matches['FTR']]
        all_results = home_results + away_results
        
        # اتساق الأداء
        home_performance = (home_matches['FTHG'] - home_matches['FTAG']).tolist()
        away_performance = (away_matches['FTAG'] - away_matches['FTHG']).tolist()
        all_performance = home_performance + away_performance
        
        return {
            'results_consistency': 1 - (np.std(all_results) / np.mean(all_results)) if len(all_results) > 1 and np.mean(all_results) > 0 else 0.5,
            'performance_consistency': 1 - (np.std(all_performance) / (np.mean(np.abs(all_performance)) + 1)) if len(all_performance) > 1 else 0.5,
            'scoring_consistency': self._calculate_scoring_consistency(home_matches, away_matches),
            'defensive_consistency': self._calculate_defensive_consistency(home_matches, away_matches),
            'form_momentum': self._calculate_enhanced_form_momentum(home_matches, away_matches),
            'volatility_index': np.std(all_results) if len(all_results) > 1 else 1.0
        }
    
    def _calculate_situational_metrics(self, home_matches, away_matches, team_name):
        """مؤشرات الظروف (منزل/خارج)"""
        home_matches_count = len(home_matches)
        away_matches_count = len(away_matches)
        
        # أداء المنزل
        home_points = (len(home_matches[home_matches['FTR'] == 'H']) * 3 + 
                      len(home_matches[home_matches['FTR'] == 'D']))
        home_win_rate = len(home_matches[home_matches['FTR'] == 'H']) / home_matches_count if home_matches_count > 0 else 0
        
        # أداء الخارج
        away_points = (len(away_matches[away_matches['FTR'] == 'A']) * 3 + 
                      len(away_matches[away_matches['FTR'] == 'D']))
        away_win_rate = len(away_matches[away_matches['FTR'] == 'A']) / away_matches_count if away_matches_count > 0 else 0
        
        return {
            'home_advantage': home_points / (home_matches_count * 3) if home_matches_count > 0 else 0,
            'away_resilience': away_points / (away_matches_count * 3) if away_matches_count > 0 else 0,
            'performance_balance': (home_win_rate + away_win_rate) / 2,
            'home_dominance': home_win_rate - away_win_rate,
            'adaptability_index': min(home_win_rate, away_win_rate) / max(home_win_rate, away_win_rate) if max(home_win_rate, away_win_rate) > 0 else 0
        }
    
    # الدوال المساعدة المحسنة
    def _calculate_advanced_xg_ratio(self, home_matches, away_matches, actual_goals):
        """نسبة xG متقدمة"""
        home_xg = (home_matches['HST'].sum() * 0.3 + home_matches['HC'].sum() * 0.02) if 'HST' in home_matches.columns else len(home_matches) * 1.5
        away_xg = (away_matches['AST'].sum() * 0.25 + away_matches['AC'].sum() * 0.015) if 'AST' in away_matches.columns else len(away_matches) * 1.2
        total_xg = home_xg + away_xg
        
        return actual_goals / total_xg if total_xg > 0 else 1.0
    
    def _calculate_overperformance_ratio(self, home_matches, away_matches, actual_goals):
        """نسبة تجاوز التوقعات"""
        expected_goals = len(home_matches) * 1.5 + len(away_matches) * 1.0
        return actual_goals / expected_goals if expected_goals > 0 else 1.0
    
    def _calculate_tactical_errors(self, home_matches, away_matches):
        """حساب الأخطاء التكتيكية"""
        home_errors = (home_matches['HF'].sum() + home_matches['HY'].sum() * 2 + home_matches['HR'].sum() * 3) if 'HF' in home_matches.columns else len(home_matches) * 15
        away_errors = (away_matches['AF'].sum() + away_matches['AY'].sum() * 2 + away_matches['AR'].sum() * 3) if 'AF' in away_matches.columns else len(away_matches) * 16
        return home_errors + away_errors
    
    def _calculate_game_management(self, home_matches, away_matches):
        """إدارة المباراة"""
        # نسبة الفوز في المباريات التي سجل فيها أولاً
        home_scored_first = home_matches[home_matches['FTHG'] > home_matches['FTAG']]
        away_scored_first = away_matches[away_matches['FTAG'] > away_matches['FTHG']]
        
        home_management = len(home_scored_first[home_scored_first['FTR'] == 'H']) / len(home_scored_first) if len(home_scored_first) > 0 else 0.5
        away_management = len(away_scored_first[away_scored_first['FTR'] == 'A']) / len(away_scored_first) if len(away_scored_first) > 0 else 0.5
        
        return (home_management + away_management) / 2
    
    def _calculate_scoring_consistency(self, home_matches, away_matches):
        """اتساق التسجيل"""
        home_goals = home_matches['FTHG'].tolist()
        away_goals = away_matches['FTAG'].tolist()
        all_goals = home_goals + away_goals
        
        if len(all_goals) < 2:
            return 0.5
        return 1 - (np.std(all_goals) / np.mean(all_goals)) if np.mean(all_goals) > 0 else 0.5
    
    def _calculate_defensive_consistency(self, home_matches, away_matches):
        """اتساق الدفاع"""
        home_conceded = home_matches['FTAG'].tolist()
        away_conceded = away_matches['FTHG'].tolist()
        all_conceded = home_conceded + away_conceded
        
        if len(all_conceded) < 2:
            return 0.5
        return 1 - (np.std(all_conceded) / np.mean(all_conceded)) if np.mean(all_conceded) > 0 else 0.5
    
    def _calculate_enhanced_form_momentum(self, home_matches, away_matches):
        """زخم محسّن"""
        # آخر 8 مباريات
        home_recent = home_matches.tail(4)
        away_recent = away_matches.tail(4)
        
        recent_points = (
            len(home_recent[home_recent['FTR'] == 'H']) * 3 +
            len(home_recent[home_recent['FTR'] == 'D']) +
            len(away_recent[away_recent['FTR'] == 'A']) * 3 +
            len(away_recent[away_recent['FTR'] == 'D'])
        )
        
        max_possible = (len(home_recent) + len(away_recent)) * 3
        return recent_points / max_possible if max_possible > 0 else 0.5
    
    def _calculate_clean_sheet_rate(self, home_matches, away_matches):
        """معدل النظافة الدفاعية"""
        home_clean = len(home_matches[home_matches['FTAG'] == 0])
        away_clean = len(away_matches[away_matches['FTHG'] == 0])
        total_matches = len(home_matches) + len(away_matches)
        
        return (home_clean + away_clean) / total_matches if total_matches > 0 else 0

    def calculate_comprehensive_score(self, metrics):
        """حساب درجة شاملة محسنة"""
        if not metrics:
            return 0
        
        weights = {
            # الأساسية (30%)
            'points_per_match': 0.10,
            'goal_difference': 0.08,
            'win_rate': 0.12,
            
            # الهجوم (25%)
            'shot_efficiency': 0.06,
            'conversion_rate': 0.06,
            'expected_goals_ratio': 0.05,
            'attacking_pressure': 0.04,
            'overperformance_ratio': 0.04,
            
            # الدفاع (20%)
            'defensive_efficiency': 0.06,
            'clean_sheet_rate': 0.06,
            'defensive_stability': 0.04,
            'disciplinary_index': 0.04,
            
            # التكتيك (10%)
            'possession_control': 0.04,
            'tactical_discipline': 0.03,
            'game_management': 0.03,
            
            # الاتساق (10%)
            'results_consistency': 0.04,
            'performance_consistency': 0.03,
            'form_momentum': 0.03,
            
            # الظروف (5%)
            'home_advantage': 0.025,
            'away_resilience': 0.025
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                # تطبيع القيم حسب كل مقياس
                normalized_value = self._normalize_metric(metric, metrics[metric])
                score += normalized_value * weight * 100
        
        return min(100, max(0, score))
    
    def _normalize_metric(self, metric_name, value):
        """تطبيع القيم حسب المقياس"""
        if value == 0:
            return 0
            
        if metric_name in ['points_per_match', 'win_rate', 'shot_efficiency', 'conversion_rate', 
                          'defensive_efficiency', 'clean_sheet_rate', 'home_advantage', 'away_resilience']:
            return min(value * 2, 1.0)  # تضخيم للقيم المنخفضة
        
        elif metric_name in ['goal_difference', 'attacking_pressure']:
            return min(value / 2, 1.0)  # تقليل للقيم العالية
        
        elif metric_name in ['expected_goals_ratio', 'overperformance_ratio']:
            return min(value, 1.5) / 1.5  # تطبيع حول 1.0
        
        else:
            return min(value, 1.0)
    
    def assess_all_teams(self):
        """تقييم جميع الفرق"""
        print("🎯 جاري تقييم جميع الفرق...")
        
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        
        for team in all_teams:
            if pd.isna(team) or team == 'nan':
                continue
                
            metrics = self.calculate_enhanced_metrics(team)
            
            if metrics:
                comprehensive_score = self.calculate_comprehensive_score(metrics)
                metrics['comprehensive_score'] = comprehensive_score
                self.teams_assessment[team] = metrics
        
        print(f"✅ تم تقييم {len(self.teams_assessment)} فريق")
        return self.teams_assessment

    def create_final_ranking(self):
        """إنشاء الترتيب النهائي للفرق"""
        if not self.teams_assessment:
            self.assess_all_teams()
        
        ranking_data = []
        for team, metrics in self.teams_assessment.items():
            row_data = {'Team': team}
            row_data.update(metrics)
            ranking_data.append(row_data)
        
        self.final_ranking = pd.DataFrame(ranking_data)
        self.final_ranking = self.final_ranking.sort_values('comprehensive_score', ascending=False)
        
        return self.final_ranking
    
    def save_results(self, output_file="comprehensive_team_ranking.csv"):
        """حفظ النتائج في ملف CSV"""
        if self.final_ranking is None:
            self.create_final_ranking()
        
        self.final_ranking.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ تم حفظ الترتيب في {output_file}")
        
        # حفظ البيانات الكاملة للمؤشرات
        full_data_file = "complete_team_metrics.csv"
        self.final_ranking.to_csv(full_data_file, index=False, encoding='utf-8-sig')
        print(f"✅ تم حفظ جميع المؤشرات في {full_data_file}")
    
    def create_comprehensive_charts(self):
        """إنشاء المنحنيات البيانية الشاملة"""
        if self.final_ranking is None:
            self.create_final_ranking()
        
        # إعداد التصميم
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. مخطط الدرجات الشاملة
        plt.subplot(2, 3, 1)
        top_20 = self.final_ranking.head(20)
        plt.barh(top_20['Team'], top_20['comprehensive_score'])
        plt.xlabel('الدرجة الشاملة')
        plt.title('أقوى 20 فريق - التقييم الشامل')
        plt.gca().invert_yaxis()
        
        # 2. مخطط مؤشرات الهجوم والدفاع
        plt.subplot(2, 3, 2)
        top_10 = self.final_ranking.head(10)
        attack_strength = top_10['shot_efficiency'] * 100
        defense_strength = top_10['defensive_efficiency'] * 100
        
        x = range(len(top_10))
        width = 0.35
        plt.bar(x, attack_strength, width, label='قوة الهجوم')
        plt.bar([i + width for i in x], defense_strength, width, label='قوة الدفاع')
        plt.xlabel('الفريق')
        plt.ylabel('النسبة المئوية')
        plt.title('قوة الهجوم vs الدفاع (أقوى 10 فرق)')
        plt.xticks([i + width/2 for i in x], top_10['Team'], rotation=45)
        plt.legend()
        
        # 3. مخطط الاتساق
        plt.subplot(2, 3, 3)
        consistency_metrics = ['scoring_consistency', 'defensive_consistency', 'results_consistency']
        consistency_data = top_10[consistency_metrics]
        
        for i, metric in enumerate(consistency_metrics):
            plt.plot(top_10['Team'], consistency_data[metric] * 100, 
                    marker='o', label=metric)
        
        plt.xlabel('الفريق')
        plt.ylabel('نسبة الاتساق %')
        plt.title('مؤشرات الاتساق للفرق')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 4. مخطط الرادار لأفضل 5 فرق
        plt.subplot(2, 3, 4)
        self._create_radar_chart(self.final_ranking.head(5))
        
        # 5. مخطط العلاقة بين الهجوم والدفاع
        plt.subplot(2, 3, 5)
        plt.scatter(self.final_ranking['shot_efficiency'] * 100, 
                   self.final_ranking['defensive_efficiency'] * 100,
                   alpha=0.6, s=50)
        plt.xlabel('كفاءة الهجوم %')
        plt.ylabel('كفاءة الدفاع %')
        plt.title('العلاقة بين كفاءة الهجوم والدفاع')
        
        # إضافة أسماء الفرق للمتطرفين
        for i, row in self.final_ranking.head(10).iterrows():
            plt.annotate(row['Team'], 
                        (row['shot_efficiency'] * 100, row['defensive_efficiency'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. مخطط توزيع الدرجات
        plt.subplot(2, 3, 6)
        plt.hist(self.final_ranking['comprehensive_score'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('الدرجة الشاملة')
        plt.ylabel('عدد الفرق')
        plt.title('توزيع الدرجات الشاملة لجميع الفرق')
        
        plt.tight_layout()
        plt.savefig('comprehensive_team_assessment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_radar_chart(self, top_5):
        """إنشاء مخطط رادار لأفضل 5 فرق"""
        categories = ['الهجوم', 'الدفاع', 'الاتساق', 'الأداء', 'القيمة']
        
        # تجميع البيانات للمخطط الرادار
        radar_data = []
        for _, team in top_5.iterrows():
            team_metrics = [
                (team['shot_efficiency'] + team['conversion_rate']) / 2 * 100,
                (team['defensive_efficiency'] + team['clean_sheet_rate']) / 2 * 100,
                (team['scoring_consistency'] + team['defensive_consistency']) / 2 * 100,
                (team['performance_balance'] + team['points_per_match'] * 10) / 2,
                team['possession_control'] * 100
            ]
            radar_data.append(team_metrics)
        
        # رسم المخطط الرادار
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # إغلاق الدائرة
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for i, (team, metrics) in enumerate(zip(top_5['Team'], radar_data)):
            metrics += metrics[:1]  # إغلاق الدائرة
            ax.plot(angles, metrics, 'o-', linewidth=2, label=team)
            ax.fill(angles, metrics, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        plt.title('مقارنة أفضل 5 فرق (مخطط رادار)', size=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return fig
    
    def generate_detailed_report(self):
        """إنشاء تقرير مفصل عن النتائج"""
        if self.final_ranking is None:
            self.create_final_ranking()
        
        print("\n" + "="*80)
        print("📊 التقرير الشامل لتقييم الفرق")
        print("="*80)
        
        print(f"\n🏆 أقوى 10 فرق بناءً على التقييم الشامل:")
        print("-" * 60)
        top_10 = self.final_ranking.head(10)
        for i, (_, team) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {team['Team']:<25} | {team['comprehensive_score']:5.1f} نقطة | "
                  f"هجوم: {team['shot_efficiency']*100:4.1f}% | دفاع: {team['defensive_efficiency']*100:4.1f}%")
        
        print(f"\n📈 الإحصائيات العامة:")
        print(f"• عدد الفرق المُقيمة: {len(self.final_ranking)}")
        print(f"• متوسط الدرجة الشاملة: {self.final_ranking['comprehensive_score'].mean():.1f}")
        print(f"• أعلى درجة: {self.final_ranking['comprehensive_score'].max():.1f}")
        print(f"• أدنى درجة: {self.final_ranking['comprehensive_score'].min():.1f}")
        
        print(f"\n🎯 الفرق المتميزة في مجالات محددة:")
        # أفضل فريق في الهجوم
        best_attack = self.final_ranking.loc[self.final_ranking['shot_efficiency'].idxmax()]
        best_defense = self.final_ranking.loc[self.final_ranking['defensive_efficiency'].idxmax()]
        most_consistent = self.final_ranking.loc[self.final_ranking['results_consistency'].idxmax()]
        
        print(f"• أفضل هجوم: {best_attack['Team']} ({best_attack['shot_efficiency']*100:.1f}%)")
        print(f"• أفضل دفاع: {best_defense['Team']} ({best_defense['defensive_efficiency']*100:.1f}%)")
        print(f"• الأكثر اتساقاً: {most_consistent['Team']} ({most_consistent['results_consistency']*100:.1f}%)")
        
        print(f"\n⚡ الفرق الأكثر تحسناً (زخم):")
        high_momentum = self.final_ranking.nlargest(5, 'form_momentum')
        for _, team in high_momentum.iterrows():
            print(f"• {team['Team']}: زخم {team['form_momentum']:.1%}")

# التشغيل الرئيسي
if __name__ == "__main__":
    # تحميل البيانات
    assessor = ComprehensiveTeamAssessment("data/football-data/combined_seasons_data.csv")
    
    # معالجة البيانات
    assessor.preprocess_data()
    
    # تقييم جميع الفرق
    assessor.assess_all_teams()
    
    # إنشاء الترتيب النهائي
    ranking = assessor.create_final_ranking()
    
    # حفظ النتائج
    assessor.save_results()
    
    # إنشاء المنحنيات البيانية
    assessor.create_comprehensive_charts()
    
    # عرض التقرير المفصل
    assessor.generate_detailed_report()
    
    print(f"\n🎉 تم الانتهاء من التقييم الشامل!")
    print(f"📁 الملفات الناتجة:")
    print(f"   • comprehensive_team_ranking.csv - الترتيب النهائي")
    print(f"   • complete_team_metrics.csv - جميع المؤشرات")
    print(f"   • comprehensive_team_assessment.png - المنحنيات البيانية")