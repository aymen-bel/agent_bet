# smart_weight/data_enricher.py 

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os

class DataEnricher:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.required_columns = [
            'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF'
        ]
        
        # قائمة ملفات المواسم المحدثة
        self.season_files = [
            "combined_seasons_data.csv",
            "england_E0_2020.csv", "england_E0_2021.csv", "england_E0_2022.csv",
            "england_E0_2023.csv", "england_E0_2024.csv", "england_E0_2025.csv"  # ⬅️ تمت الإضافة
        ]
    
    def load_and_combine_data(self) -> pd.DataFrame:
        """تحميل ودمج البيانات من جميع الملفات - الإصدار المصحح"""
        all_data = []
        
        try:
            # تحميل جميع ملفات المواسم
            for filename in self.season_files:
                file_path = os.path.join(self.data_path, filename)
                if os.path.exists(file_path):
                    try:
                        df_season = pd.read_csv(file_path)
                        # إضافة عمود الموسم للملفات الفردية
                        if filename.startswith('england_E0_'):
                            year = filename.split('_')[-1].split('.')[0]
                            df_season['Season'] = year
                        all_data.append(df_season)
                        print(f"✅ تم تحميل: {filename} ({len(df_season)} مباراة)")
                    except Exception as e:
                        print(f"⚠️  خطأ في تحميل {filename}: {e}")
                        continue
            
            if not all_data:
                print("❌ لم يتم العثور على أي ملفات بيانات")
                return self._create_sample_data()
            
            # دمج جميع البيانات
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"📊 إجمالي المباريات المدموجة: {len(combined_data)}")
            
            return combined_data
            
        except Exception as e:
            print(f"❌ خطأ في تحميل البيانات: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """إنشاء بيانات نموذجية عند عدم توفر البيانات الحقيقية"""
        print("🔄 إنشاء بيانات نموذجية للتدريب...")
        
        teams = [
            'Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Tottenham',
            'Man United', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace'
        ]
        
        sample_data = []
        for i in range(200):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            match_data = {
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': np.random.poisson(1.5),
                'FTAG': np.random.poisson(1.2),
                'HS': np.random.randint(8, 20),
                'AS': np.random.randint(6, 18),
                'HST': np.random.randint(2, 8),
                'AST': np.random.randint(1, 7),
                'Season': '2024'
            }
            sample_data.append(match_data)
        
        return pd.DataFrame(sample_data)
    
    def enrich_match_data(self, matches_data) -> List[Dict]:
        """إثراء بيانات المباريات - الإصدار المصحح تماماً"""
        try:
            # إذا كان DataFrame، تحويله إلى list of dicts أولاً
            if hasattr(matches_data, 'to_dict'):
                matches_data = matches_data.to_dict('records')
            
            enriched_matches = []
            
            for match in matches_data:
                # التحقق من نوع البيانات
                if not isinstance(match, dict):
                    print(f"⚠️  تخطي عنصر غير قاموس: {type(match)}")
                    continue
                    
                enriched_match = self._enrich_single_match(match)
                if enriched_match:
                    enriched_matches.append(enriched_match)
            
            print(f"✅ تم إثراء {len(enriched_matches)} مباراة")
            return enriched_matches
            
        except Exception as e:
            print(f"❌ خطأ في إثراء البيانات: {e}")
            # في حالة الخطأ، نعود البيانات الأصلية
            if hasattr(matches_data, 'to_dict'):
                return matches_data.to_dict('records')
            return matches_data if isinstance(matches_data, list) else []
    
    def _enrich_single_match(self, match: Dict) -> Dict:
        """إثراء مباراة واحدة - الإصدار المصحح"""
        try:
            # التحقق من وجود البيانات الأساسية
            home_team = match.get('HomeTeam')
            away_team = match.get('AwayTeam')
            
            if not home_team or not away_team:
                print(f"⚠️  مباراة ناقصة البيانات: {home_team} vs {away_team}")
                return None
            
            # استخراج البيانات الأساسية مع قيم افتراضية
            home_goals = self._safe_int_get(match, 'FTHG', 0)
            away_goals = self._safe_int_get(match, 'FTAG', 0)
            
            # حساب المقاييس الأساسية
            result = self._calculate_match_result(home_goals, away_goals)
            total_goals = home_goals + away_goals
            goal_difference = home_goals - away_goals
            
            # إضافة المقاييس المحسوبة
            enriched_match = match.copy()
            enriched_match.update({
                'result': result,
                'total_goals': total_goals,
                'goal_difference': goal_difference,
                'home_attack_strength': self._calculate_attack_strength(home_goals, self._safe_int_get(match, 'HS', 1)),
                'away_attack_strength': self._calculate_attack_strength(away_goals, self._safe_int_get(match, 'AS', 1)),
                'home_defense_strength': self._calculate_defense_strength(away_goals, self._safe_int_get(match, 'AS', 1)),
                'away_defense_strength': self._calculate_defense_strength(home_goals, self._safe_int_get(match, 'HS', 1)),
            })
            
            return enriched_match
            
        except Exception as e:
            print(f"⚠️  خطأ في إثراء المباراة: {e}")
            return match
    
    def _safe_int_get(self, data_dict: Dict, key: str, default: int = 0) -> int:
        """استخراج قيمة رقمية بأمان من القاموس"""
        try:
            value = data_dict.get(key, default)
            if value is None:
                return default
            return int(float(value))  # تحويل إلى float أولاً ثم int للتعامل مع القيم النصية
        except (ValueError, TypeError):
            return default
    
    def _calculate_match_result(self, home_goals: int, away_goals: int) -> str:
        """حساب نتيجة المباراة"""
        if home_goals > away_goals:
            return 'H'
        elif away_goals > home_goals:
            return 'A'
        else:
            return 'D'
    
    def _calculate_attack_strength(self, goals: int, shots: int) -> float:
        """حساب قوة الهجوم"""
        try:
            if shots == 0:
                return 0.0
            efficiency = goals / shots
            return min(efficiency * 10, 1.0)  # تطبيع بين 0 و 1
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_defense_strength(self, goals_conceded: int, opponent_shots: int) -> float:
        """حساب قوة الدفاع"""
        try:
            if opponent_shots == 0:
                return 1.0  # إذا لم توجد تسديدات، فالدفاع مثالي
            efficiency = goals_conceded / opponent_shots
            return max(1.0 - (efficiency * 10), 0.0)  # تطبيع بين 0 و 1
        except (ValueError, TypeError):
            return 1.0
    
    def prepare_training_data(self, enriched_data: List[Dict]) -> List[Dict]:
        """تحضير بيانات التدريب - الإصدار المصحح والمحسن"""
        training_data = []
        
        for match in enriched_data:
            try:
                # التحقق من أن العنصر قاموس
                if not isinstance(match, dict):
                    continue
                
                # استخراج الفرق
                home_team = match.get('HomeTeam', 'Unknown')
                away_team = match.get('AwayTeam', 'Unknown')
                
                # حساب المقاييس بناءً على البيانات الفعلية
                home_goals = self._safe_int_get(match, 'FTHG', 0)
                away_goals = self._safe_int_get(match, 'FTAG', 0)
                
                # مقاييس الفريق المضيف
                team_metrics = {
                    'points_per_match': self._calculate_points(home_goals, away_goals, is_home=True),
                    'win_rate': 1.0 if home_goals > away_goals else 0.0,
                    'goal_difference': home_goals - away_goals,
                    'goals_per_match': home_goals,
                    'conceded_per_match': away_goals,
                    'current_form': np.random.uniform(0.2, 0.9),  # سيتم حسابه من البيانات التاريخية
                    'home_advantage': 1.15,
                    'defensive_efficiency': self._calculate_defense_strength(
                        away_goals, self._safe_int_get(match, 'AS', 1)
                    ),
                    'motivation_factor': self._calculate_motivation(home_team, home_goals, away_goals)
                }
                
                # مقاييس الفريق الضيف
                opponent_metrics = {
                    'points_per_match': self._calculate_points(away_goals, home_goals, is_home=False),
                    'win_rate': 1.0 if away_goals > home_goals else 0.0,
                    'goal_difference': away_goals - home_goals,
                    'goals_per_match': away_goals,
                    'conceded_per_match': home_goals,
                    'current_form': np.random.uniform(0.2, 0.9),
                    'away_advantage': 0.85,
                    'defensive_efficiency': self._calculate_defense_strength(
                        home_goals, self._safe_int_get(match, 'HS', 1)
                    ),
                    'motivation_factor': self._calculate_motivation(away_team, away_goals, home_goals)
                }
                
                context = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_advantage': team_metrics['home_advantage'],
                    'match_importance': self._calculate_match_importance(home_team, away_team),
                    'season_stage': np.random.uniform(0.2, 0.8),
                    'current_motivation': (team_metrics['motivation_factor'] + opponent_metrics['motivation_factor']) / 2,
                    'injury_impact': np.random.uniform(0.7, 1.0)
                }
                
                # النتيجة الفعلية
                actual_result = {
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'match_played': True
                }
                
                training_data.append({
                    'team_metrics': team_metrics,
                    'opponent_metrics': opponent_metrics,
                    'context': context,
                    'actual_result': actual_result
                })
                
            except Exception as e:
                print(f"⚠️  خطأ في تحضير بيانات التدريب للمباراة: {e}")
                continue
        
        print(f"✅ تم تحضير {len(training_data)} عينة تدريب")
        return training_data
    
    def _calculate_points(self, goals_for: int, goals_against: int, is_home: bool) -> float:
        """حساب النقاط من نتيجة المباراة"""
        if goals_for > goals_against:
            return 3.0
        elif goals_for == goals_against:
            return 1.0
        else:
            return 0.0
    
    def _calculate_motivation(self, team: str, goals_for: int, goals_against: int) -> float:
        """حساب الحافز بناءً على الأداء"""
        goal_difference = goals_for - goals_against
        
        if goal_difference >= 2:
            return 1.3  # فوز كبير
        elif goal_difference == 1:
            return 1.1  # فوز بفارق هدف
        elif goal_difference == 0:
            return 1.0  # تعادل
        elif goal_difference == -1:
            return 0.9  # خسارة بفارق هدف
        else:
            return 0.7  # خسارة كبيرة
    
    def _calculate_match_importance(self, home_team: str, away_team: str) -> float:
        """حساب أهمية المباراة بناءً على الفريقين"""
        big_teams = ['Man City', 'Liverpool', 'Arsenal', 'Chelsea', 'Man United', 'Tottenham']
        
        if home_team in big_teams and away_team in big_teams:
            return 1.5  # مباراة قمة
        elif home_team in big_teams or away_team in big_teams:
            return 1.2  # مباراة مهمة
        else:
            return 1.0  # مباراة عادية

    def get_current_season_2025_data(self) -> List[Dict]:
        """الحصول على بيانات الموسم الحالي 2025 من المصادر الحقيقية"""
        try:
            current_file = os.path.join(self.data_path, "england_E0_2025.csv")
            if os.path.exists(current_file):
                df_2025 = pd.read_csv(current_file)
                print(f"✅ تم تحميل بيانات الموسم الحالي 2025: {len(df_2025)} مباراة")
                return self.enrich_match_data(df_2025)
            else:
                print("⚠️  ملف الموسم الحالي 2025 غير موجود، استخدام البيانات المحاكاة")
                return self._get_simulated_2025_data()
                
        except Exception as e:
            print(f"❌ خطأ في تحميل بيانات 2025: {e}")
            return self._get_simulated_2025_data()
    
    def _get_simulated_2025_data(self) -> List[Dict]:
        """إنشاء بيانات محاكاة للموسم الحالي 2025"""
        current_season_matches = []
        teams_2025 = [
            'Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Tottenham',
            'Man United', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace',
            'Wolves', 'Aston Villa', 'Everton', 'Southampton', 'Leicester',
            'Fulham', 'Brentford', 'Nottm Forest', 'Bournemouth', 'Ipswich'
        ]
        
        for i in range(100):
            home_team = np.random.choice(teams_2025)
            away_team = np.random.choice([t for t in teams_2025 if t != home_team])
            
            match = {
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': np.random.poisson(1.5),
                'FTAG': np.random.poisson(1.2),
                'HS': np.random.randint(8, 20),
                'AS': np.random.randint(6, 18),
                'HST': np.random.randint(2, 8),
                'AST': np.random.randint(1, 7),
                'Season': '2025'
            }
            
            current_season_matches.append(match)
        
        print(f"✅ تم إنشاء {len(current_season_matches)} مباراة محاكاة للموسم 2025")
        return current_season_matches

# ==================== فئة مساعدة للتحقق من البيانات ====================

class DataValidator:
    """فئة مساعدة للتحقق من صحة البيانات"""
    
    @staticmethod
    def validate_match_structure(match_data: Dict) -> bool:
        """التحقق من هيكل بيانات المباراة"""
        required_fields = ['team_metrics', 'opponent_metrics', 'context']
        
        for field in required_fields:
            if field not in match_data:
                print(f"❌ حقل مفقود: {field}")
                return False
        
        # التحقق من الحقول الداخلية
        if not isinstance(match_data['team_metrics'], dict):
            print("❌ team_metrics يجب أن يكون قاموس")
            return False
            
        if not isinstance(match_data['context'], dict):
            print("❌ context يجب أن يكون قاموس")
            return False
        
        return True
    
    @staticmethod
    def fix_missing_metrics(match_data: Dict) -> Dict:
        """إصلاح المقاييس المفقودة في بيانات المباراة"""
        fixed_data = match_data.copy()
        
        # إصلاح team_metrics
        if 'team_metrics' not in fixed_data or not fixed_data['team_metrics']:
            fixed_data['team_metrics'] = DataValidator._get_default_metrics()
        
        # إصلاح opponent_metrics
        if 'opponent_metrics' not in fixed_data or not fixed_data['opponent_metrics']:
            fixed_data['opponent_metrics'] = DataValidator._get_default_metrics()
        
        # إصلاح context
        if 'context' not in fixed_data or not fixed_data['context']:
            fixed_data['context'] = {
                'home_team': 'Unknown',
                'away_team': 'Unknown',
                'match_date': '2025-01-01'
            }
        
        return fixed_data
    
    @staticmethod
    def _get_default_metrics() -> Dict:
        """الحصول على مقاييس افتراضية"""
        return {
            'points_per_match': 1.5,
            'win_rate': 0.5,
            'goal_difference': 0,
            'goals_per_match': 1.5,
            'conceded_per_match': 1.5,
            'current_form': 0.5,
            'home_advantage': 1.1,
            'defensive_efficiency': 0.7,
            'motivation_factor': 1.0
        }