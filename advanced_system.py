# advanced_system.py - النسخة المحسنة مع التكامل الكامل للتنبؤ والرهان
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data.future_fixtures_loader import FutureFixturesLoader
from models.neural_predictor import NeuralWeightPredictor
from models.genetic_optimizer import GeneticWeightOptimizer
from models.confidence_calibrator import ConfidenceCalibrator
from smart_weight.smart_optimizer import SmartWeightOptimizer
from smart_weight.data_enricher import DataEnricher
from smart_weight.performance_evaluator import PerformanceEvaluator
from output.output_manager import OutputManager

# ==================== استيراد مكونات التنبؤ ====================
from predict.advanced_predictor import AdvancedMatchPredictor
from predict.realistic_predictor import RealisticMatchPredictor
from predict.enhanced_ml_predictor import EnhancedMLPredictor

# ==================== استيراد المكونات الجديدة ====================
from team_assessor.advanced_team_assessor import EnhancedTeamAssessor
from betting_engine.advanced_predictor import AdvancedBettingPredictor
from betting_engine.betting_types import BettingEngine, BetType, MarketType
from data_validator.data_validator import DataValidator

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import requests
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== مساعد JSON المتقدم ====================

class AdvancedJSONEncoder(json.JSONEncoder):
    """مشفر JSON متقدم يدعم أنواع numpy والتواريخ"""
    
    def default(self, obj):
        try:
            # معالجة أنواع numpy
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            
            # للمتجهات والقوائم المعقدة
            return super().default(obj)
        except Exception as e:
            print(f"⚠️  تحذير في ترميز JSON: {e}")
            return str(obj)

def safe_json_serialize(obj, **kwargs):
    """دالة آمنة لتحويل الكائنات إلى JSON"""
    return json.dumps(obj, cls=AdvancedJSONEncoder, **kwargs)

def safe_json_dump(obj, file_path, **kwargs):
    """حفظ آمن للكائنات في ملف JSON"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(safe_json_serialize(obj, **kwargs))

# ==================== نظام البيانات الزمني المتكامل المحسن ====================

class TemporalDataIntegrator:
    """مدمج بيانات زمني للمواسم المتعاقبة - نسخة محسنة"""
    
    def __init__(self):
        self.season_transitions = {
            '2020-2021': {'promoted': [], 'relegated': []},
            '2021-2022': {'promoted': [], 'relegated': []},
            '2022-2023': {'promoted': [], 'relegated': []},
            '2023-2024': {'promoted': [], 'relegated': []},
            '2024-2025': {
                'promoted': ['Leicester City', 'Ipswich Town', 'Southampton'],
                'relegated': ['Luton Town', 'Burnley', 'Sheffield United']
            }
        }
        
        # قوة الفرق التاريخية عبر المواسم
        self.team_strength_evolution = self._initialize_strength_evolution()
    
    def _initialize_strength_evolution(self):
        """تهيئة تطور قوة الفرق عبر المواسم"""
        base_strengths_2024 = {
            'Arsenal': 2.1, 'Aston Villa': 1.6, 'Bournemouth': 1.0, 'Brentford': 1.2,
            'Brighton': 1.5, 'Chelsea': 1.8, 'Crystal Palace': 1.1, 'Everton': 0.9,
            'Fulham': 1.1, 'Liverpool': 2.2, 'Manchester City': 2.4, 'Manchester United': 1.7,
            'Newcastle United': 1.6, 'Nottingham Forest': 1.0, 'Tottenham': 1.8,
            'West Ham United': 1.4, 'Wolves': 1.1,
            # الفرق الصاعدة
            'Leicester City': 1.3, 'Ipswich Town': 0.8, 'Southampton': 1.0
        }
        
        evolution = {}
        for team, strength in base_strengths_2024.items():
            evolution[team] = {
                '2020-2021': max(0.5, strength * 0.7),
                '2021-2022': max(0.5, strength * 0.8),
                '2022-2023': max(0.5, strength * 0.9),
                '2023-2024': strength,
                '2024-2025': strength
            }
        
        return evolution
    
    def get_team_strength_for_season(self, team: str, season: str) -> float:
        """الحصول على قوة الفريق لموسم محدد"""
        if team in self.team_strength_evolution:
            return self.team_strength_evolution[team].get(season, 1.0)
        return 1.0  # قيمة افتراضية للفرق الجديدة
    
    def calculate_team_adaptation(self, team: str, is_newly_promoted: bool) -> float:
        """حساب عامل تكيف الفريق مع الدوري"""
        if is_newly_promoted:
            return 0.85  # الفرق الصاعدة تحتاج وقت للتكيف
        return 1.0
    
    def get_seasonal_adjustment(self, current_season: str, data_season: str) -> float:
        """معامل تعديل للبيانات التاريخية بناءً على قدمها"""
        try:
            season_difference = int(current_season[:4]) - int(data_season[:4])
            decay_factor = 0.8  # تضاؤل أهمية البيانات القديمة
            return decay_factor ** season_difference
        except:
            return 0.7

# ==================== الرزنامة الرسمية لموسم 2025/2026 المحسنة ====================

class PremierLeagueCalendar2025Real:
    def __init__(self):
        self.season_start = datetime(2025, 8, 9)
        self.season_end = datetime(2026, 5, 24)
        
        # فرق الدوري الإنجليزي 2025/2026
        self.teams_2025 = self.load_current_teams()
        
        # تحميل الرزنامة الكاملة من API
        self.all_fixtures = self.load_complete_fixtures()
        self.match_weeks = self.organize_fixtures_by_week()
    
    def load_complete_fixtures(self) -> List[Dict]:
        """تحميل الرزنامة الكاملة من API مع البيانات المحلية"""
        try:
            # استخدام API الجديد
            loader = FutureFixturesLoader()
            api_fixtures = loader.load_fixtures(use_cache=True)
            
            if api_fixtures:
                print(f"✅ تم تحميل {len(api_fixtures)} مباراة من API")
                return self._ensure_fixtures_serializable(api_fixtures)
            else:
                raise Exception("فشل تحميل البيانات من API")
                
        except Exception as e:
            print(f"⚠️  خطأ في تحميل البيانات من API: {e}")
            return self.load_fallback_fixtures()
    
    def _ensure_fixtures_serializable(self, fixtures: List[Dict]) -> List[Dict]:
        """ضمان إمكانية تسلسل البيانات إلى JSON"""
        serializable_fixtures = []
        for fixture in fixtures:
            try:
                # تحويل أي قيم غير قابلة للتسلسل
                safe_fixture = {}
                for key, value in fixture.items():
                    if isinstance(value, (np.integer, np.int32, np.int64)):
                        safe_fixture[key] = int(value)
                    elif isinstance(value, (np.floating, np.float32, np.float64)):
                        safe_fixture[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        safe_fixture[key] = value.tolist()
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        safe_fixture[key] = value.isoformat()
                    else:
                        safe_fixture[key] = value
                serializable_fixtures.append(safe_fixture)
            except Exception as e:
                print(f"⚠️  خطأ في معالجة المباراة: {e}")
                continue
        return serializable_fixtures
    
    def load_fallback_fixtures(self) -> List[Dict]:
        """تحميل البيانات الاحتياطية"""
        fixtures = []
        
        try:
            # محاولة تحميل من الملفات المحلية
            local_files = [
                "data/seasons/england_E0_2025.csv",
                "data/seasons/premier_league_2025_fixtures.csv",
                "data/seasons/premier_league_2025_fixtures.json"
            ]
            
            for file_path in local_files:
                if os.path.exists(file_path):
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        fixtures.extend(self.convert_df_to_fixtures(df))
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            fixtures.extend(data)
                    break
            
            if not fixtures:
                fixtures = self.generate_fallback_fixtures()
                
        except Exception as e:
            print(f"⚠️  خطأ في تحميل البيانات الاحتياطية: {e}")
            fixtures = self.generate_fallback_fixtures()
        
        return self._ensure_fixtures_serializable(fixtures)
    
    def convert_df_to_fixtures(self, df: pd.DataFrame) -> List[Dict]:
        """تحويل DataFrame إلى تنسيق الفيكستشر"""
        fixtures = []
        
        for _, row in df.iterrows():
            try:
                # معالجة التاريخ والوقت
                match_date = self.parse_date(row.get('Date', ''))
                is_played = pd.notna(row.get('FTHG', None))
                
                fixture = {
                    'MatchID': str(hash(f"{row['HomeTeam']}_{row['AwayTeam']}_{row.get('Date', '')}")),
                    'Date': match_date.strftime('%d/%m/%Y') if match_date else 'Unknown',
                    'Time': '15:00',  # وقت افتراضي
                    'HomeTeam': str(row['HomeTeam']),
                    'AwayTeam': str(row['AwayTeam']),
                    'HomeGoals': int(row.get('FTHG', 0)) if is_played else None,
                    'AwayGoals': int(row.get('FTAG', 0)) if is_played else None,
                    'Status': 'FINISHED' if is_played else 'SCHEDULED',
                    'Matchday': int(self.calculate_matchday(match_date)) if match_date else 1,
                    'Venue': f"{row['HomeTeam']} Stadium",
                    'Referee': 'Unknown',
                    'IsPlayed': bool(is_played),
                    'Season': '2025-2026',
                    'API_Data': False
                }
                fixtures.append(fixture)
                
            except Exception as e:
                print(f"⚠️  خطأ في تحويل المباراة: {e}")
                continue
        
        return fixtures
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """تحليل تنسيقات التاريخ المختلفة"""
        try:
            # تنسيق DD/MM/YYYY
            if '/' in date_str:
                return datetime.strptime(date_str, '%d/%m/%Y')
            # تنسيق YYYY-MM-DD
            elif '-' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                return None
        except:
            return None
    
    def calculate_matchday(self, match_date: datetime) -> int:
        """حساب يوم المباراة بناءً على التاريخ"""
        if not match_date:
            return 1
        days_from_start = (match_date - self.season_start).days
        return max(1, (days_from_start // 7) + 1)
    
    def generate_fallback_fixtures(self) -> List[Dict]:
        """إنشاء فيكستشر احتياطية"""
        print("🔄 إنشاء رزنامة احتياطية...")
        loader = FutureFixturesLoader()
        fixtures = loader.generate_simulated_fixtures()
        return self._ensure_fixtures_serializable(fixtures)
    
    def organize_fixtures_by_week(self) -> Dict:
        """تنظيم المباريات حسب الأسابيع"""
        match_weeks = {}
        
        for fixture in self.all_fixtures:
            try:
                week_num = int(fixture.get('Matchday', 1))
                
                if week_num not in match_weeks:
                    match_weeks[week_num] = {
                        'week_number': week_num,
                        'start_date': self.get_week_start_date(week_num),
                        'matches': []
                    }
                
                match_data = {
                    'home_team': str(fixture['HomeTeam']),
                    'away_team': str(fixture['AwayTeam']),
                    'match_date': fixture['Date'],
                    'match_time': fixture['Time'],
                    'venue': fixture['Venue'],
                    'week_number': week_num,
                    'actual_result': {
                        'home_goals': int(fixture.get('HomeGoals', 0)) if fixture.get('HomeGoals') is not None else None,
                        'away_goals': int(fixture.get('AwayGoals', 0)) if fixture.get('AwayGoals') is not None else None,
                        'match_played': bool(fixture.get('IsPlayed', False))
                    },
                    'fixture_data': fixture
                }
                
                match_weeks[week_num]['matches'].append(match_data)
                
            except Exception as e:
                print(f"⚠️  خطأ في تنظيم المباراة: {e}")
                continue
        
        return match_weeks
    
    def get_week_start_date(self, week_num: int) -> str:
        """الحصول على تاريخ بداية الأسبوع"""
        start_date = self.season_start + timedelta(days=(week_num-1)*7)
        return start_date.strftime('%Y-%m-%d')
    
    def load_current_teams(self):
        """تحميل قائمة الفرق الحالية"""
        teams = [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
            'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham',
            'West Ham United', 'Wolverhampton Wanderers'
        ]
        return teams
    
    def get_played_matches(self) -> List[Dict]:
        """الحصول على المباريات التي لعبت بالفعل"""
        played_matches = []
        
        for week_num, week_data in self.match_weeks.items():
            for match in week_data['matches']:
                if match.get('actual_result') and match['actual_result'].get('match_played'):
                    match_data = match.copy()
                    match_data['week_number'] = week_num
                    played_matches.append(match_data)
        
        return played_matches
    
    def get_upcoming_matches_with_results(self, weeks_ahead: int = 4) -> List[Dict]:
        """الحصول على المباريات القادمة مع النتائج السابقة"""
        current_week = self.get_current_week()
        upcoming_matches = []
        
        for week in range(current_week, min(current_week + weeks_ahead + 1, 39)):
            if week in self.match_weeks:
                week_matches = self.match_weeks[week]['matches']
                for match in week_matches:
                    if not match['actual_result']['match_played']:
                        match_data = match.copy()
                        match_data['week_number'] = week
                        
                        # إضافة الإحصائيات التاريخية
                        match_data['team_history'] = self._get_team_season_history(match['home_team'])
                        match_data['opponent_history'] = self._get_team_season_history(match['away_team'])
                        
                        upcoming_matches.append(match_data)
        
        return upcoming_matches
    
    def _get_team_season_history(self, team: str) -> Dict:
        """الحصول على تاريخ الفريق في الموسم الحالي"""
        played = self.get_played_matches()
        team_matches = [m for m in played if m['home_team'] == team or m['away_team'] == team]
        
        if not team_matches:
            return {
                'matches_played': 0,
                'points': 0,
                'goals_for': 0,
                'goals_against': 0,
                'form': []
            }
        
        points = 0
        goals_for = 0
        goals_against = 0
        form = []
        
        for match in team_matches:
            if match['home_team'] == team:
                home_goals = match['actual_result']['home_goals'] or 0
                away_goals = match['actual_result']['away_goals'] or 0
                goals_for += home_goals
                goals_against += away_goals
                if home_goals > away_goals:
                    points += 3
                    form.append('W')
                elif home_goals == away_goals:
                    points += 1
                    form.append('D')
                else:
                    form.append('L')
            else:
                away_goals = match['actual_result']['away_goals'] or 0
                home_goals = match['actual_result']['home_goals'] or 0
                goals_for += away_goals
                goals_against += home_goals
                if away_goals > home_goals:
                    points += 3
                    form.append('W')
                elif away_goals == home_goals:
                    points += 1
                    form.append('D')
                else:
                    form.append('L')
        
        return {
            'matches_played': len(team_matches),
            'points': points,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_difference': goals_for - goals_against,
            'form': form[-5:],  # آخر 5 مباريات
            'points_per_match': float(points / len(team_matches)) if team_matches else 0.0
        }
    
    def get_current_week(self) -> int:
        """الحصول على الأسبوع الحالي بناءً على التاريخ الفعلي"""
        today = datetime.now()
        if today < self.season_start:
            return 0
        elif today > self.season_end:
            return 39
        
        days_elapsed = (today - self.season_start).days
        return min((days_elapsed // 7) + 1, 38)

# ==================== محرك الرهان المحسن ====================

class EnhancedBettingEngine:
    """محرك رهان محسن مع معالجة أفضل للأخطاء"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_1x2_odds(self, home_prob: float, draw_prob: float, away_prob: float) -> Dict[str, float]:
        """توليد odds لرهانات 1X2 مع معالجة الأخطاء"""
        try:
            # التحقق من القيم الصفرية
            if home_prob <= 0 or draw_prob <= 0 or away_prob <= 0:
                self.logger.warning("⚠️ احتمالات تحتوي على قيم صفرية، استخدام قيم افتراضية")
                home_prob = max(0.01, home_prob)
                draw_prob = max(0.01, draw_prob)
                away_prob = max(0.01, away_prob)
            
            # تطبيع الاحتمالات
            total_prob = home_prob + draw_prob + away_prob
            if total_prob <= 0:
                home_prob, draw_prob, away_prob = 0.33, 0.34, 0.33
                total_prob = 1.0
            
            home_prob /= total_prob
            draw_prob /= total_prob
            away_prob /= total_prob
            
            # هامش الربح للكتاب (5%)
            margin = 0.05
            
            # حساب odds العادلة
            fair_home_odds = 1.0 / home_prob
            fair_draw_odds = 1.0 / draw_prob
            fair_away_odds = 1.0 / away_prob
            
            # تطبيق هامش الربح
            home_odds = fair_home_odds * (1 - margin)
            draw_odds = fair_draw_odds * (1 - margin)
            away_odds = fair_away_odds * (1 - margin)
            
            return {
                '1': max(1.1, min(10.0, home_odds)),
                'X': max(1.1, min(10.0, draw_odds)),
                '2': max(1.1, min(10.0, away_odds))
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds 1X2: {e}")
            # إرجاع قيم افتراضية آمنة
            return {'1': 2.0, 'X': 3.0, '2': 2.5}
    
    def generate_goals_odds(self, over_prob: float, under_prob: float, line: float = 2.5) -> Dict[str, float]:
        """توليد odds لرهانات over/under"""
        try:
            # التحقق من القيم الصفرية
            if over_prob <= 0 or under_prob <= 0:
                over_prob = max(0.01, over_prob)
                under_prob = max(0.01, under_prob)
            
            # تطبيع الاحتمالات
            total_prob = over_prob + under_prob
            if total_prob <= 0:
                over_prob, under_prob = 0.5, 0.5
                total_prob = 1.0
            
            over_prob /= total_prob
            under_prob /= total_prob
            
            margin = 0.05
            
            fair_over_odds = 1.0 / over_prob
            fair_under_odds = 1.0 / under_prob
            
            over_odds = fair_over_odds * (1 - margin)
            under_odds = fair_under_odds * (1 - margin)
            
            return {
                'over': max(1.1, min(5.0, over_odds)),
                'under': max(1.1, min(5.0, under_odds))
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds الأهداف: {e}")
            return {'over': 1.8, 'under': 1.9}
    
    def generate_btts_odds(self, yes_prob: float, no_prob: float) -> Dict[str, float]:
        """توليد odds لرهانات BTTS"""
        try:
            if yes_prob <= 0 or no_prob <= 0:
                yes_prob = max(0.01, yes_prob)
                no_prob = max(0.01, no_prob)
            
            total_prob = yes_prob + no_prob
            if total_prob <= 0:
                yes_prob, no_prob = 0.5, 0.5
                total_prob = 1.0
            
            yes_prob /= total_prob
            no_prob /= total_prob
            
            margin = 0.05
            
            fair_yes_odds = 1.0 / yes_prob
            fair_no_odds = 1.0 / no_prob
            
            yes_odds = fair_yes_odds * (1 - margin)
            no_odds = fair_no_odds * (1 - margin)
            
            return {
                'yes': max(1.1, min(5.0, yes_odds)),
                'no': max(1.1, min(5.0, no_odds))
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في توليد odds BTTS: {e}")
            return {'yes': 1.8, 'no': 1.9}

# ==================== مدير الإخراج المتكامل المحسن ====================

class OutputManager:
    """مدير متكامل محسن لحفظ جميع مخرجات النظام"""
    
    def __init__(self):
        self.base_output_dir = "output"
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """إنشاء هيكل المجلدات المطلوب"""
        directories = [
            "data_loading",
            "historical_data", 
            "merged_data",
            "training",
            "reinforcement_learning",
            "predictions",
            "team_assessor",
            "betting_engine", 
            "data_validator",
            "training/generations",
            "training/models",
            "reinforcement_learning/episodes",
            "predictions/detailed",
            "predictions/summary",
            "predictions/comparison",
            "predictions/performance",
            "team_assessor/assessments",
            "team_assessor/reports",
            "betting_engine/predictions",
            "betting_engine/performance",
            "betting_engine/bet_types",
            "data_validator/validation_reports",
            "data_validator/accuracy_analysis"
        ]
        
        for directory in directories:
            os.makedirs(f"{self.base_output_dir}/{directory}", exist_ok=True)
        
        print("📁 تم إنشاء هيكل مجلدات الإخراج")
    
    def save_data_loading_report(self, fixtures: List[Dict], source: str, load_type: str = "fixtures"):
        """حفظ تقرير تحميل البيانات"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            'load_timestamp': datetime.now().isoformat(),
            'data_type': load_type,
            'data_source': source,
            'total_count': len(fixtures),
            'future_matches': len([f for f in fixtures if f.get('IsFuture', False)]),
            'played_matches': len([f for f in fixtures if f.get('IsPlayed', False)]),
            'sample_fixtures': fixtures[:10]  # عينة من البيانات
        }
        
        json_filename = f"{self.base_output_dir}/data_loading/{load_type}_report_{timestamp}.json"
        self.save_json(report_data, json_filename)
        
        # حفظ ملخص CSV
        if fixtures and load_type == "fixtures":
            csv_data = []
            for fixture in fixtures:
                csv_data.append({
                    'MatchID': fixture.get('MatchID'),
                    'Date': fixture.get('Date'),
                    'HomeTeam': fixture.get('HomeTeam'),
                    'AwayTeam': fixture.get('AwayTeam'),
                    'Status': fixture.get('Status'),
                    'IsFuture': fixture.get('IsFuture', False),
                    'IsPlayed': fixture.get('IsPlayed', False),
                    'Week': fixture.get('Matchday', 0)
                })
            
            df = pd.DataFrame(csv_data)
            csv_filename = f"{self.base_output_dir}/data_loading/fixtures_summary_{timestamp}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        print(f"💾 تم حفظ تقرير التحميل: {json_filename}")
        return json_filename
    
    def save_historical_data(self, historical_data: pd.DataFrame, source_files: List[str]):
        """حفظ البيانات التاريخية المجمعة"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # حفظ البيانات المجمعة
        combined_filename = f"{self.base_output_dir}/historical_data/combined_seasons_{timestamp}.csv"
        historical_data.to_csv(combined_filename, index=False)
        
        # حفظ تقرير المصادر
        source_report = {
            'combined_timestamp': datetime.now().isoformat(),
            'total_matches': len(historical_data),
            'source_files': source_files,
            'seasons_covered': historical_data['Season'].unique().tolist() if 'Season' in historical_data.columns else [],
            'teams_count': len(historical_data['HomeTeam'].unique()) if 'HomeTeam' in historical_data.columns else 0,
            'data_columns': list(historical_data.columns)
        }
        
        report_filename = f"{self.base_output_dir}/historical_data/sources_report_{timestamp}.json"
        self.save_json(source_report, report_filename)
        
        print(f"💾 تم حفظ البيانات التاريخية: {combined_filename}")
        return combined_filename
    
    def save_merged_training_data(self, merged_data: List[Dict], metadata: Dict):
        """حفظ بيانات التدريب المتكاملة"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # حفظ البيانات المدمجة
        data_filename = f"{self.base_output_dir}/merged_data/merged_training_data_{timestamp}.json"
        self.save_json({
            'metadata': metadata,
            'training_data': merged_data[:100]  # حفظ عينة فقط لأغراض التحليل
        }, data_filename)
        
        # حفظ إحصائيات الدمج
        stats = {
            'total_samples': len(merged_data),
            'historical_samples': len([d for d in merged_data if d.get('season_context', {}).get('is_historical', True)]),
            'current_season_samples': len([d for d in merged_data if not d.get('season_context', {}).get('is_historical', True)]),
            'merge_timestamp': datetime.now().isoformat(),
            'feature_count': len(merged_data[0]['team_metrics']) if merged_data else 0
        }
        
        stats_filename = f"{self.base_output_dir}/merged_data/merge_stats_{timestamp}.json"
        self.save_json(stats, stats_filename)
        
        print(f"💾 تم حفظ البيانات المدمجة: {data_filename}")
        return data_filename
    
    def save_training_generation(self, generation: int, best_fitness: float, 
                               avg_fitness: float, population_stats: Dict, weights: Dict):
        """حفظ بيانات كل جيل تدريب"""
        gen_data = {
            'generation_number': int(generation),
            'timestamp': datetime.now().isoformat(),
            'best_fitness': float(best_fitness),
            'average_fitness': float(avg_fitness),
            'population_size': int(population_stats.get('population_size', 0)),
            'mutation_rate': float(population_stats.get('mutation_rate', 0)),
            'crossover_rate': float(population_stats.get('crossover_rate', 0)),
            'best_weights_sample': {k: float(v) for i, (k, v) in enumerate(weights.items()) if i < 10}  # عينة من الأوزان
        }
        
        # حفظ كل جيل في ملف منفصل
        gen_filename = f"{self.base_output_dir}/training/generations/generation_{generation:03d}.json"
        self.save_json(gen_data, gen_filename)
        
        # تحديث ملف التلخيص
        self.update_training_summary(gen_data)
        
        return gen_filename
    
    def update_training_summary(self, generation_data: Dict):
        """تحديث ملف تلخيص التدريب"""
        summary_file = f"{self.base_output_dir}/training/training_summary.json"
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {'generations': [], 'start_time': datetime.now().isoformat()}
        
        summary['generations'].append(generation_data)
        summary['last_update'] = datetime.now().isoformat()
        summary['total_generations'] = len(summary['generations'])
        
        self.save_json(summary, summary_file)
    
    def save_rl_episode(self, episode: int, reward: float, exploration_rate: float,
                       actions_taken: List, state_history: List):
        """حفظ بيانات كل حلقة تعلم معزز"""
        episode_data = {
            'episode_number': int(episode),
            'timestamp': datetime.now().isoformat(),
            'reward_achieved': float(reward),
            'exploration_rate': float(exploration_rate),
            'total_actions': len(actions_taken),
            'actions_sample': actions_taken[:10],  # عينة من الإجراءات
            'state_history_sample': state_history[:5]  # عينة من الحالات
        }
        
        episode_filename = f"{self.base_output_dir}/reinforcement_learning/episodes/episode_{episode:03d}.json"
        self.save_json(episode_data, episode_filename)
        
        self.update_rl_summary(episode_data)
        
        return episode_filename
    
    def update_rl_summary(self, episode_data: Dict):
        """تحديث ملف تلخيص التعلم المعزز"""
        summary_file = f"{self.base_output_dir}/reinforcement_learning/rl_summary.json"
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {'episodes': [], 'start_time': datetime.now().isoformat()}
        
        summary['episodes'].append(episode_data)
        summary['last_update'] = datetime.now().isoformat()
        summary['total_episodes'] = len(summary['episodes'])
        
        self.save_json(summary, summary_file)
    
    def save_prediction_batch(self, predictions: List[Dict], batch_type: str = "current_season"):
        """حفظ دفعة من التنبؤات"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # حفظ تنبؤات مفصلة
        detailed_filename = f"{self.base_output_dir}/predictions/detailed/{batch_type}_predictions_{timestamp}.json"
        self.save_json({
            'batch_type': batch_type,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'predictions': predictions
        }, detailed_filename)
        
        # حفظ ملخص تنبؤات ك CSV
        summary_data = []
        for pred in predictions:
            match_info = pred.get('match_info', {})
            prediction_data = pred.get('prediction', {}).get('predictions', [{}])[0] if pred.get('prediction', {}).get('predictions') else {}
            
            summary_data.append({
                'match_date': match_info.get('match_date', 'Unknown'),
                'home_team': match_info.get('home_team', 'Unknown'),
                'away_team': match_info.get('away_team', 'Unknown'),
                'predicted_home_goals': float(prediction_data.get('home_goals', 0)),
                'predicted_away_goals': float(prediction_data.get('away_goals', 0)),
                'confidence': float(prediction_data.get('confidence', 0)),
                'week_number': int(match_info.get('week_number', 0)),
                'timestamp': pred.get('generated_at', 'Unknown')
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_filename = f"{self.base_output_dir}/predictions/summary/{batch_type}_summary_{timestamp}.csv"
            df.to_csv(summary_filename, index=False, encoding='utf-8')
        else:
            summary_filename = None
        
        print(f"💾 تم حفظ {len(predictions)} تنبؤ في: {detailed_filename}")
        return detailed_filename, summary_filename

    def save_prediction_comparison(self, comparison_data: Dict, filename_suffix: str = ""):
        """حفظ مقارنة أداء نماذج التنبؤ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        
        filename = f"{self.base_output_dir}/predictions/comparison/model_comparison{suffix}_{timestamp}.json"
        
        self.save_json(comparison_data, filename)
        
        print(f"📊 تم حفظ مقارنة النماذج: {filename}")
        return filename

    def save_prediction_performance(self, performance_data: Dict, model_name: str):
        """حفظ أداء التنبؤ لنموذج محدد"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/predictions/performance/{model_name}_performance_{timestamp}.json"
        
        self.save_json(performance_data, filename)
        
        print(f"📈 تم حفظ أداء النموذج {model_name}: {filename}")
        return filename

    def save_team_assessment(self, team_assessment: Dict, assessment_type: str = "comprehensive"):
        """حفظ تقييم الفرق"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/team_assessor/assessments/{assessment_type}_assessment_{timestamp}.json"
        
        self.save_json(team_assessment, filename)
        
        print(f"🏆 تم حفظ تقييم الفرق: {filename}")
        return filename

    def save_team_ranking(self, ranking_data: pd.DataFrame, ranking_type: str = "final"):
        """حفظ ترتيب الفرق"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/team_assessor/reports/{ranking_type}_ranking_{timestamp}.csv"
        ranking_data.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"📊 تم حفظ ترتيب الفرق: {filename}")
        return filename

    def save_betting_predictions(self, betting_predictions: Dict, prediction_type: str = "comprehensive"):
        """حفظ تنبؤات الرهان"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/betting_engine/predictions/{prediction_type}_betting_{timestamp}.json"
        
        self.save_json(betting_predictions, filename)
        
        print(f"🎯 تم حفظ تنبؤات الرهان: {filename}")
        return filename

    def save_betting_performance(self, performance_data: Dict, performance_type: str = "overall"):
        """حفظ أداء الرهان"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/betting_engine/performance/{performance_type}_performance_{timestamp}.json"
        
        self.save_json(performance_data, filename)
        
        print(f"💰 تم حفظ أداء الرهان: {filename}")
        return filename

    def save_betting_types_analysis(self, bet_analysis: Dict, analysis_type: str = "comprehensive"):
        """حفظ تحليل أنواع الرهان"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/betting_engine/bet_types/{analysis_type}_analysis_{timestamp}.json"
        
        self.save_json(bet_analysis, filename)
        
        print(f"🎰 تم حفظ تحليل أنواع الرهان: {filename}")
        return filename

    def save_validation_report(self, validation_report: Dict, validation_type: str = "comprehensive"):
        """حفظ تقرير التحقق من البيانات"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/data_validator/validation_reports/{validation_type}_validation_{timestamp}.json"
        
        self.save_json(validation_report, filename)
        
        print(f"🔍 تم حفظ تقرير التحقق: {filename}")
        return filename

    def save_accuracy_analysis(self, accuracy_data: Dict, analysis_type: str = "model_performance"):
        """حفظ تحليل الدقة"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.base_output_dir}/data_validator/accuracy_analysis/{analysis_type}_{timestamp}.json"
        
        self.save_json(accuracy_data, filename)
        
        print(f"📈 تم حفظ تحليل الدقة: {filename}")
        return filename

    def save_json(self, data: Dict, filepath: str):
        """حفظ البيانات في ملف JSON"""
        try:
            # إنشاء المجلد إذا لم يكن موجوداً
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"❌ خطأ في حفظ الملف {filepath}: {e}")

# ==================== مدير الموسم الحالي المحسن بشكل كامل ====================

class EnhancedCurrentSeasonManager:
    def __init__(self):
        self.temporal_integrator = TemporalDataIntegrator()
        self.calendar = PremierLeagueCalendar2025Real()
        self.current_season = "2025-2026"
        self.current_data = self.load_current_season_data()
        
    def load_current_season_data(self):
        """تحميل بيانات الموسم الحالي 2025/2026"""
        try:
            current_file = "data/seasons/england_E0_2025.csv"
            df = pd.read_csv(current_file)
            
            # تحويل إلى التنسيق الداخلي للنظام
            processed_data = self.process_raw_season_data(df, "2025")
            return processed_data
            
        except Exception as e:
            print(f"❌ خطأ في تحميل الموسم الحالي: {e}")
            return self.create_fallback_current_data()
    
    def process_raw_season_data(self, df, season_year):
        """معالجة البيانات الخام إلى تنسيق النظام"""
        matches_data = []
        
        for _, row in df.iterrows():
            try:
                match_data = {
                    'team_metrics': self.extract_team_metrics(row, True),  # فريق本地
                    'opponent_metrics': self.extract_team_metrics(row, False),  # فريق الضيف
                    'context': {
                        'home_team': str(row['HomeTeam']),
                        'away_team': str(row['AwayTeam']),
                        'match_date': str(row['Date']),
                        'season': f"{season_year}-{int(season_year)+1}",
                        'is_current_season': True
                    },
                    'actual_result': {
                        'home_goals': int(row.get('FTHG', 0)) if pd.notna(row.get('FTHG')) else None,
                        'away_goals': int(row.get('FTAG', 0)) if pd.notna(row.get('FTAG')) else None,
                        'match_played': bool(pd.notna(row.get('FTHG')))
                    }
                }
                matches_data.append(match_data)
                
            except Exception as e:
                print(f"⚠️ خطأ في معالجة مباراة: {e}")
                continue
                
        return matches_data
    
    def extract_team_metrics(self, match_row, is_home_team):
        """استخراج مقاييس الفريق من بيانات المباراة"""
        team = str(match_row['HomeTeam'] if is_home_team else match_row['AwayTeam'])
        
        return {
            'points_per_match': float(self.calculate_ppm_from_history(team)),
            'win_rate': float(self.calculate_win_rate_from_history(team)),
            'goals_per_match': float(self.calculate_gpm_from_history(team, is_home_team)),
            'conceded_per_match': float(self.calculate_cpm_from_history(team, is_home_team)),
            'current_form': float(self.get_current_form(team)),
            'home_advantage': 1.15 if is_home_team else 0.85,
            'defensive_efficiency': 0.7,
            'motivation_factor': 1.0
        }
    
    def calculate_ppm_from_history(self, team):
        """حساب متوسط النقاط من التاريخ (مؤقت)"""
        return np.random.uniform(1.0, 2.5)
    
    def calculate_win_rate_from_history(self, team):
        """حساب معدل الفوز من التاريخ (مؤقت)"""
        return np.random.uniform(0.3, 0.8)
    
    def calculate_gpm_from_history(self, team, is_home):
        """حساب متوسط الأهداف من التاريخ (مؤقت)"""
        return np.random.uniform(0.8, 2.5)
    
    def calculate_cpm_from_history(self, team, is_home):
        """حساب متوسط الأهداف المستلمة من التاريخ (مؤقت)"""
        return np.random.uniform(0.8, 2.2)
    
    def get_current_form(self, team):
        """الحصول على الأداء الحالي (مؤقت)"""
        return np.random.uniform(0.2, 0.9)
    
    def create_fallback_current_data(self):
        """إنشاء بيانات احتياطية للموسم الحالي"""
        return []
        
    def get_current_season_context(self, team_metrics: Dict, opponent_metrics: Dict, 
                                 context: Dict) -> Dict:
        """إثراء البيانات بسياق الموسم الحالي"""
        enriched_context = context.copy()
        
        # إضافة معلومات الموسم الحالي
        current_week = self.calendar.get_current_week()
        team_history = self.calendar._get_team_season_history(context['home_team'])
        opponent_history = self.calendar._get_team_season_history(context['away_team'])
        
        enriched_context.update({
            'current_week': current_week,
            'season_stage': float(current_week / 38.0),
            'team_current_season': team_history,
            'opponent_current_season': opponent_history,
            'is_newly_promoted_team': self._is_newly_promoted(context['home_team']),
            'is_newly_promoted_opponent': self._is_newly_promoted(context['away_team']),
            'adaptation_factor': self._calculate_adaptation_factor(context['home_team'], context['away_team'])
        })
        
        return enriched_context
    
    def _is_newly_promoted(self, team: str) -> bool:
        """التحقق إذا كان فريقاً صاعداً حديثاً"""
        promoted_teams = ['Leicester City', 'Ipswich Town', 'Southampton']
        return team in promoted_teams
    
    def _calculate_adaptation_factor(self, home_team: str, away_team: str) -> float:
        """حساب عامل التكيف للفرق الصاعدة"""
        home_adaptation = 0.85 if self._is_newly_promoted(home_team) else 1.0
        away_adaptation = 0.80 if self._is_newly_promoted(away_team) else 1.0
        
        return float((home_adaptation + away_adaptation) / 2)
    
    def validate_and_enrich_match_data(self, match_data: Dict) -> Dict:
        """التحقق من البيانات وإثرائها ببيانات الموسم الحالي"""
        try:
            # التحقق من البيانات الأساسية
            if not self._validate_basic_data(match_data):
                return self._create_fallback_data(match_data)
            
            # إثراء ببيانات الموسم الحالي
            enriched_data = match_data.copy()
            context = match_data.get('context', {})
            
            # إضافة سياق الموسم الحالي
            enriched_context = self.get_current_season_context(
                match_data.get('team_metrics', {}),
                match_data.get('opponent_metrics', {}),
                context
            )
            
            enriched_data['context'] = enriched_context
            
            # إضافة عوامل التكيف الموسمي
            enriched_data['seasonal_factors'] = {
                'team_adaptation': float(self.temporal_integrator.calculate_team_adaptation(
                    context.get('home_team', ''), 
                    self._is_newly_promoted(context.get('home_team', ''))
                )),
                'opponent_adaptation': float(self.temporal_integrator.calculate_team_adaptation(
                    context.get('away_team', ''),
                    self._is_newly_promoted(context.get('away_team', ''))
                )),
                'historical_weight': float(self._calculate_historical_weight(context))
            }
            
            return enriched_data
            
        except Exception as e:
            logging.error(f"❌ خطأ في إثراء بيانات المباراة: {e}")
            return self._create_fallback_data(match_data)
    
    def _validate_basic_data(self, match_data: Dict) -> bool:
        """التحقق من البيانات الأساسية"""
        required_fields = ['team_metrics', 'opponent_metrics', 'context']
        for field in required_fields:
            if field not in match_data or not match_data[field]:
                return False
        
        context = match_data['context']
        if not context.get('home_team') or not context.get('away_team'):
            return False
            
        return True
    
    def _create_fallback_data(self, match_data: Dict) -> Dict:
        """إنشاء بيانات احتياطية في حالة فشل التحقق"""
        base_data = match_data.copy()
        
        # استخدام بيانات الموسم الحالي إذا كانت متاحة
        context = base_data.get('context', {})
        home_team = context.get('home_team', 'Unknown')
        away_team = context.get('away_team', 'Unknown')
        
        home_history = self.calendar._get_team_season_history(home_team)
        away_history = self.calendar._get_team_season_history(away_team)
        
        base_data.setdefault('team_metrics', {})
        base_data.setdefault('opponent_metrics', {})
        base_data.setdefault('context', {})
        
        # تحديث المقاييس بناءً على الأداء الحالي
        base_data['team_metrics'].update({
            'current_form': float(len([f for f in home_history.get('form', []) if f == 'W']) / 5.0),
            'points_per_match': float(home_history.get('points_per_match', 1.0)),
            'goals_per_match': float(home_history.get('goals_for', 0) / max(home_history.get('matches_played', 1), 1)),
            'conceded_per_match': float(home_history.get('goals_against', 0) / max(home_history.get('matches_played', 1), 1))
        })
        
        base_data['opponent_metrics'].update({
            'current_form': float(len([f for f in away_history.get('form', []) if f == 'W']) / 5.0),
            'points_per_match': float(away_history.get('points_per_match', 1.0)),
            'goals_per_match': float(away_history.get('goals_for', 0) / max(away_history.get('matches_played', 1), 1)),
            'conceded_per_match': float(away_history.get('goals_against', 0) / max(away_history.get('matches_played', 1), 1))
        })
        
        return base_data
    
    def _calculate_historical_weight(self, context: Dict) -> float:
        """حساب وزن البيانات التاريخية بناءً على سياق الموسم"""
        current_week = self.calendar.get_current_week()
        
        if current_week <= 5:
            return 0.3  # الاعتماد القليل على التاريخ في بداية الموسم
        elif current_week <= 15:
            return 0.6  # زيادة الاعتماد مع تقدم الموسم
        else:
            return 0.8  # اعتماد كبير على التاريخ في نهاية الموسم

# ==================== نظام التنبؤ المتكامل المحسن ====================

class IntegratedPredictionEngine:
    """محرك تنبؤ متكامل محسن يجمع جميع نماذج التنبؤ"""
    
    def __init__(self, team_assessment_data: Dict):
        self.team_data = team_assessment_data
        self.betting_engine = EnhancedBettingEngine()
        
        # تهيئة جميع نماذج التنبؤ
        self.advanced_predictor = AdvancedMatchPredictor(team_assessment_data)
        self.realistic_predictor = RealisticMatchPredictor(team_assessment_data)
        self.ml_predictor = EnhancedMLPredictor()
        
        self.prediction_history = {}
        self.model_performance = {}
        
    def generate_comprehensive_prediction(self, home_team: str, away_team: str, 
                                        venue: str = "home", external_factors: Dict = None) -> Dict:
        """توليد تنبؤ شامل باستخدام جميع النماذج"""
        
        if home_team not in self.team_data or away_team not in self.team_data:
            return self._get_fallback_prediction(home_team, away_team)
        
        predictions = {}
        
        try:
            # التنبؤ باستخدام النموذج المتقدم (بواسون)
            predictions['advanced'] = self.advanced_predictor.predict_match(
                home_team, away_team, venue
            )
            
            # التنبؤ باستخدام النموذج الواقعي
            predictions['realistic'] = self.realistic_predictor.predict_match(
                home_team, away_team, venue, external_factors
            )
            
            # توليد تنبؤ إجماعي
            consensus_prediction = self._generate_consensus_prediction(predictions)
            
            # حساب أداء النماذج
            model_scores = self._calculate_model_scores(predictions, consensus_prediction)
            
            # توليد تنبؤات الرهان
            betting_predictions = self._generate_betting_predictions(consensus_prediction, predictions)
            
            result = {
                'home_team': home_team,
                'away_team': away_team,
                'venue': venue,
                'predictions': predictions,
                'consensus_prediction': consensus_prediction,
                'betting_predictions': betting_predictions,
                'model_scores': model_scores,
                'recommendations': self._generate_recommendations(consensus_prediction, model_scores, betting_predictions),
                'timestamp': datetime.now().isoformat()
            }
            
            # حفظ في السجل
            match_id = f"{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.prediction_history[match_id] = result
            
            return result
            
        except Exception as e:
            print(f"❌ خطأ في التنبؤ الشامل: {e}")
            return self._get_fallback_prediction(home_team, away_team)
    
    def _generate_betting_predictions(self, consensus_prediction: Dict, model_predictions: Dict) -> Dict:
        """توليد تنبؤات الرهان من التنبؤات الأساسية"""
        try:
            home_goals = consensus_prediction.get('home_goals', 1)
            away_goals = consensus_prediction.get('away_goals', 1)
            confidence = consensus_prediction.get('confidence', 0.5)
            
            # حساب الاحتمالات للرهانات المختلفة
            total_goals = home_goals + away_goals
            
            # احتمالات 1X2 مبسطة
            if home_goals > away_goals:
                home_win_prob = min(0.9, confidence * 0.8)
                draw_prob = max(0.05, (1 - confidence) * 0.3)
                away_win_prob = max(0.05, (1 - confidence) * 0.2)
            elif home_goals < away_goals:
                home_win_prob = max(0.05, (1 - confidence) * 0.2)
                draw_prob = max(0.05, (1 - confidence) * 0.3)
                away_win_prob = min(0.9, confidence * 0.8)
            else:
                home_win_prob = max(0.05, (1 - confidence) * 0.3)
                draw_prob = min(0.9, confidence * 0.7)
                away_win_prob = max(0.05, (1 - confidence) * 0.3)
            
            # تطبيع الاحتمالات
            total_1x2 = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total_1x2
            draw_prob /= total_1x2
            away_win_prob /= total_1x2
            
            # توليد odds
            odds_1x2 = self.betting_engine.generate_1x2_odds(home_win_prob, draw_prob, away_win_prob)
            
            # احتمالات BTTS
            btts_prob = 0.5 if home_goals > 0 and away_goals > 0 else 0.3
            btts_odds = self.betting_engine.generate_btts_odds(btts_prob, 1 - btts_prob)
            
            # احتمالات over/under
            over_prob = 0.6 if total_goals > 2.5 else 0.4
            goals_odds = self.betting_engine.generate_goals_odds(over_prob, 1 - over_prob)
            
            betting_predictions = {
                '1x2_market': {
                    'home_win': {'probability': home_win_prob, 'odds': odds_1x2.get('1', 2.0)},
                    'draw': {'probability': draw_prob, 'odds': odds_1x2.get('X', 3.0)},
                    'away_win': {'probability': away_win_prob, 'odds': odds_1x2.get('2', 2.5)}
                },
                'double_chance': {
                    '1X': {'probability': home_win_prob + draw_prob, 'odds': 1.5},
                    'X2': {'probability': draw_prob + away_win_prob, 'odds': 1.5},
                    '12': {'probability': home_win_prob + away_win_prob, 'odds': 1.2}
                },
                'btts': {
                    'yes': {'probability': btts_prob, 'odds': btts_odds.get('yes', 1.8)},
                    'no': {'probability': 1 - btts_prob, 'odds': btts_odds.get('no', 1.9)}
                },
                'over_under': {
                    'over_2.5': {'probability': over_prob, 'odds': goals_odds.get('over', 2.0)},
                    'under_2.5': {'probability': 1 - over_prob, 'odds': goals_odds.get('under', 1.8)}
                },
                'correct_score': {
                    f"{home_goals}-{away_goals}": {'probability': confidence, 'odds': 8.0}
                },
                'recommended_bets': self._generate_recommended_bets(
                    home_win_prob, draw_prob, away_win_prob, btts_prob, over_prob, confidence
                )
            }
            
            return betting_predictions
            
        except Exception as e:
            print(f"❌ خطأ في توليد تنبؤات الرهان: {e}")
            return {}
    
    def _generate_recommended_bets(self, home_win_prob: float, draw_prob: float, away_win_prob: float,
                                 btts_prob: float, over_2_5_prob: float, confidence: float) -> List[Dict]:
        """توليد رهانات موصى بها بناءً على الاحتمالات"""
        recommended_bets = []
        
        # تحديد أفضل الرهانات بناءً على الاحتمالات والثقة
        if home_win_prob > 0.6 and confidence > 0.6:
            recommended_bets.append({
                'type': '1x2',
                'selection': 'home_win',
                'confidence': home_win_prob * confidence,
                'stake_recommendation': 'medium'
            })
        
        if away_win_prob > 0.6 and confidence > 0.6:
            recommended_bets.append({
                'type': '1x2',
                'selection': 'away_win',
                'confidence': away_win_prob * confidence,
                'stake_recommendation': 'medium'
            })
        
        if btts_prob > 0.7:
            recommended_bets.append({
                'type': 'btts',
                'selection': 'yes',
                'confidence': btts_prob,
                'stake_recommendation': 'low'
            })
        
        if over_2_5_prob > 0.65:
            recommended_bets.append({
                'type': 'over_under',
                'selection': 'over_2.5',
                'confidence': over_2_5_prob,
                'stake_recommendation': 'medium'
            })
        
        if draw_prob > 0.4 and confidence > 0.5:
            recommended_bets.append({
                'type': 'double_chance',
                'selection': '1X' if home_win_prob > away_win_prob else 'X2',
                'confidence': draw_prob * confidence,
                'stake_recommendation': 'low'
            })
        
        # ترتيب الرهانات حسب الثقة
        recommended_bets.sort(key=lambda x: x['confidence'], reverse=True)
        return recommended_bets
    
    def _generate_consensus_prediction(self, predictions: Dict) -> Dict:
        """توليد تنبؤ إجماعي من جميع النماذج"""
        try:
            # جمع جميع التنبؤات
            all_predictions = []
            
            # من النموذج المتقدم
            if predictions.get('advanced'):
                adv = predictions['advanced']
                if 'score_prediction' in adv:
                    all_predictions.append({
                        'home_goals': int(adv['score_prediction']['home_goals']),
                        'away_goals': int(adv['score_prediction']['away_goals']),
                        'confidence': 0.7,
                        'model': 'advanced'
                    })
            
            # من النموذج الواقعي
            if predictions.get('realistic'):
                real = predictions['realistic']
                if 'multiple_predictions' in real and real['multiple_predictions']:
                    best_pred = real['multiple_predictions'][0]
                    all_predictions.append({
                        'home_goals': int(best_pred['home_goals']),
                        'away_goals': int(best_pred['away_goals']),
                        'confidence': float(best_pred.get('confidence', 0.6)),
                        'model': 'realistic'
                    })
            
            if not all_predictions:
                return {
                    'home_goals': 1,
                    'away_goals': 1,
                    'confidence': 0.5,
                    'consensus_type': 'fallback'
                }
            
            # حساب المتوسط المرجح
            total_confidence = sum(p['confidence'] for p in all_predictions)
            weighted_home = sum(p['home_goals'] * p['confidence'] for p in all_predictions) / total_confidence
            weighted_away = sum(p['away_goals'] * p['confidence'] for p in all_predictions) / total_confidence
            
            # التقريب لنتيجة واقعية
            home_goals = max(0, int(round(weighted_home)))
            away_goals = max(0, int(round(weighted_away)))
            
            # حساب الثقة الإجمالية
            avg_confidence = float(total_confidence / len(all_predictions))
            
            return {
                'home_goals': home_goals,
                'away_goals': away_goals,
                'confidence': avg_confidence,
                'consensus_type': 'weighted_average',
                'models_used': len(all_predictions),
                'detailed_scores': all_predictions
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإجماع: {e}")
            return {
                'home_goals': 1,
                'away_goals': 1,
                'confidence': 0.5,
                'consensus_type': 'error_fallback'
            }
    
    def _calculate_model_scores(self, predictions: Dict, consensus: Dict) -> Dict:
        """حساب درجات أداء النماذج"""
        scores = {}
        
        try:
            for model_name, prediction in predictions.items():
                if not prediction:
                    scores[model_name] = {'score': 0, 'reason': 'no_prediction'}
                    continue
                
                # حساب الانحراف عن الإجماع
                if model_name == 'advanced' and 'score_prediction' in prediction:
                    model_home = int(prediction['score_prediction']['home_goals'])
                    model_away = int(prediction['score_prediction']['away_goals'])
                elif model_name == 'realistic' and 'multiple_predictions' in prediction:
                    if prediction['multiple_predictions']:
                        model_home = int(prediction['multiple_predictions'][0]['home_goals'])
                        model_away = int(prediction['multiple_predictions'][0]['away_goals'])
                    else:
                        scores[model_name] = {'score': 0, 'reason': 'no_predictions'}
                        continue
                else:
                    scores[model_name] = {'score': 0, 'reason': 'invalid_format'}
                    continue
                
                consensus_home = int(consensus['home_goals'])
                consensus_away = int(consensus['away_goals'])
                
                # حساب درجة التشابه (كلما قل الانحراف زادت الدرجة)
                home_diff = abs(model_home - consensus_home)
                away_diff = abs(model_away - consensus_away)
                total_diff = home_diff + away_diff
                
                # تحويل إلى درجة من 0-1 (0 = أسوأ, 1 = أفضل)
                similarity_score = float(max(0, 1 - (total_diff * 0.2)))
                
                scores[model_name] = {
                    'score': similarity_score,
                    'deviation': total_diff,
                    'details': {
                        'model_prediction': f"{model_home}-{model_away}",
                        'consensus': f"{consensus_home}-{consensus_away}",
                        'home_difference': home_diff,
                        'away_difference': away_diff
                    }
                }
            
            return scores
            
        except Exception as e:
            print(f"❌ خطأ في حساب درجات النماذج: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, consensus: Dict, model_scores: Dict, betting_predictions: Dict) -> List[str]:
        """توليد توصيات بناءً على التنبؤ الإجماعي وتنبؤات الرهان"""
        recommendations = []
        
        home_goals = int(consensus['home_goals'])
        away_goals = int(consensus['away_goals'])
        confidence = float(consensus['confidence'])
        
        # التوصية الأساسية
        if home_goals > away_goals:
            recommendations.append(f"📈 فوز متوقع للفريق المنزل: {home_goals}-{away_goals}")
        elif away_goals > home_goals:
            recommendations.append(f"📉 فوز متوقع للفريق الضيف: {away_goals}-{home_goals}")
        else:
            recommendations.append(f"⚖️ تعادل متوقع: {home_goals}-{away_goals}")
        
        # توصيات بناءً على مستوى الثقة
        if confidence >= 0.8:
            recommendations.append("✅ ثقة عالية في التنبؤ - يمكن الاعتماد عليه")
        elif confidence >= 0.6:
            recommendations.append("⚠️ ثقة متوسطة - يوصى بالتحقق من العوامل الإضافية")
        else:
            recommendations.append("🔴 ثقة منخفضة - التنبؤ غير مؤكد")
        
        # توصيات بناءً على نتيجة المباراة المتوقعة
        total_goals = home_goals + away_goals
        if total_goals >= 4:
            recommendations.append("⚽ مباراة عالية الأهداف متوقعة")
        elif total_goals <= 1:
            recommendations.append("🛡️ مباراة منخفضة الأهداف متوقعة")
        
        # توصيات بناءً على أداء النماذج
        if 'advanced' in model_scores and 'realistic' in model_scores:
            adv_score = float(model_scores['advanced'].get('score', 0))
            real_score = float(model_scores['realistic'].get('score', 0))
            
            if abs(adv_score - real_score) > 0.3:
                better_model = 'advanced' if adv_score > real_score else 'realistic'
                recommendations.append(f"🎯 النموذج {better_model} أكثر توافقاً مع الإجماع")
        
        # توصيات الرهان
        if betting_predictions and 'recommended_bets' in betting_predictions:
            top_bets = betting_predictions['recommended_bets'][:2]  # أفضل رهانين
            for bet in top_bets:
                bet_type = bet.get('type', '').upper()
                selection = bet.get('selection', '')
                bet_confidence = bet.get('confidence', 0)
                
                if bet_confidence > 0.6:
                    recommendations.append(f"🎰 رهان موصى به: {bet_type} - {selection} (ثقة: {bet_confidence:.1%})")
        
        return recommendations
    
    def _get_fallback_prediction(self, home_team: str, away_team: str) -> Dict:
        """تنبؤ احتياطي في حالة فشل النماذج"""
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': {'fallback': 'used'},
            'consensus_prediction': {
                'home_goals': 1,
                'away_goals': 1,
                'confidence': 0.3,
                'consensus_type': 'fallback'
            },
            'betting_predictions': {},
            'model_scores': {'fallback': {'score': 0, 'reason': 'fallback_used'}},
            'recommendations': ['⚠️ استخدام التنبؤ الاحتياطي due to model errors'],
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_prediction_accuracy(self, actual_results: Dict) -> Dict:
        """تقييم دقة التنبؤات مقابل النتائج الفعلية"""
        evaluation = {
            'total_predictions': 0,
            'correct_score_predictions': 0,
            'correct_result_predictions': 0,
            'model_breakdown': {},
            'overall_accuracy': 0,
            'details': []
        }
        
        for match_id, prediction_data in self.prediction_history.items():
            if match_id in actual_results:
                actual = actual_results[match_id]
                consensus = prediction_data.get('consensus_prediction', {})
                
                if consensus and 'home_goals' in consensus and 'away_goals' in consensus:
                    evaluation['total_predictions'] += 1
                    
                    # التحقق من صحة تنبؤ النتيجة
                    pred_home = int(consensus['home_goals'])
                    pred_away = int(consensus['away_goals'])
                    actual_home = int(actual.get('home_goals', 0))
                    actual_away = int(actual.get('away_goals', 0))
                    
                    # دقة النتيجة بالضبط
                    if pred_home == actual_home and pred_away == actual_away:
                        evaluation['correct_score_predictions'] += 1
                    
                    # دقة نتيجة المباراة (فوز/تعادل/خسارة)
                    pred_result = 'H' if pred_home > pred_away else 'A' if pred_away > pred_home else 'D'
                    actual_result = 'H' if actual_home > actual_away else 'A' if actual_away > actual_home else 'D'
                    
                    if pred_result == actual_result:
                        evaluation['correct_result_predictions'] += 1
                    
                    # حفظ التفاصيل
                    evaluation['details'].append({
                        'match_id': match_id,
                        'predicted': f"{pred_home}-{pred_away}",
                        'actual': f"{actual_home}-{actual_away}",
                        'score_correct': pred_home == actual_home and pred_away == actual_away,
                        'result_correct': pred_result == actual_result,
                        'confidence': float(consensus.get('confidence', 0))
                    })
        
        # حساب النسب المئوية
        if evaluation['total_predictions'] > 0:
            evaluation['score_accuracy'] = float(evaluation['correct_score_predictions'] / evaluation['total_predictions'])
            evaluation['result_accuracy'] = float(evaluation['correct_result_predictions'] / evaluation['total_predictions'])
            evaluation['overall_accuracy'] = evaluation['result_accuracy']  # استخدام دقة النتيجة كمقياس رئيسي
        
        return evaluation

# ==================== النظام الرئيسي المحسن بشكل كامل ====================

class EnhancedBettingSystem2025:
    def __init__(self, data_path: str = "data/football-data"):
        self.data_path = data_path
        self.data_enricher = DataEnricher(data_path)
        self.smart_optimizer = SmartWeightOptimizer()
        self.neural_predictor = NeuralWeightPredictor()
        self.genetic_optimizer = GeneticWeightOptimizer()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # المكونات الأساسية
        self.season_manager = EnhancedCurrentSeasonManager()
        self.output_manager = OutputManager()  # مدير الإخراج الجديد
        self.temporal_integrator = TemporalDataIntegrator()
        
        # محرك التنبؤ المتكامل
        self.prediction_engine = None
        self.team_assessment_data = {}
        
        self.is_trained = False
        self.training_history = []
        self.optimal_weights = None
        self.performance_metrics = {}
        
        # إعداد التسجيل
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_all_components(self):
        """تهيئة جميع مكونات النظام"""
        self.logger.info("🔧 تهيئة جميع مكونات النظام...")
        
        try:
            # تحميل بيانات تقييم الفرق
            self.team_assessment_data = self._load_team_assessment_data()
            
            # تهيئة محرك التنبؤ المتكامل
            self.prediction_engine = IntegratedPredictionEngine(self.team_assessment_data)
            
            self.logger.info("✅ تم تهيئة جميع مكونات النظام")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ فشل في تهيئة المكونات: {e}")
            return False
    
    def _load_team_assessment_data(self) -> Dict:
        """تحميل بيانات تقييم الفرق"""
        team_data = {}
        
        try:
            # استخدام بيانات من الموسم الحالي
            teams = self.season_manager.calendar.teams_2025
            
            for team in teams:
                team_history = self.season_manager.calendar._get_team_season_history(team)
                
                # حساب المقاييس الأساسية
                matches_played = max(team_history.get('matches_played', 1), 1)
                goals_for = team_history.get('goals_for', 0)
                goals_against = team_history.get('goals_against', 0)
                points = team_history.get('points', 0)
                form = team_history.get('form', [])
                
                team_data[team] = {
                    'goals_per_match': float(goals_for / matches_played),
                    'goals_conceded_per_match': float(goals_against / matches_played),
                    'shot_efficiency': float(np.random.uniform(0.08, 0.15)),  # قيمة افتراضية
                    'conversion_rate': float(np.random.uniform(0.1, 0.2)),    # قيمة افتراضية
                    'defensive_efficiency': float(max(0.3, 1.0 - (goals_against / (matches_played * 2)))),
                    'clean_sheet_rate': float(len([f for f in form if 'W' in f or 'D' in f]) / max(len(form), 1)),
                    'home_strength': 1.1,  # قيمة افتراضية
                    'away_strength': 0.9,   # قيمة افتراضية
                    'points_per_match': float(points / matches_played),
                    'current_form': float(len([f for f in form[-5:] if f == 'W']) / 5.0) if len(form) >= 5 else 0.5,
                    'consistency_score': float(np.random.uniform(0.6, 0.9)),  # قيمة افتراضية
                    'comprehensive_score': float((points / matches_played) * 25)  # تحويل إلى مقياس 0-100 تقريباً
                }
            
            self.logger.info(f"📊 تم تحميل بيانات تقييم لـ {len(team_data)} فريق")
            return team_data
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل بيانات التقييم: {e}")
            return self._create_fallback_team_data()
    
    def _create_fallback_team_data(self) -> Dict:
        """إنشاء بيانات تقييم افتراضية للفرق"""
        fallback_data = {}
        teams = self.season_manager.calendar.teams_2025
        
        for team in teams:
            fallback_data[team] = {
                'goals_per_match': float(np.random.uniform(1.0, 2.5)),
                'goals_conceded_per_match': float(np.random.uniform(0.8, 2.2)),
                'shot_efficiency': float(np.random.uniform(0.08, 0.15)),
                'conversion_rate': float(np.random.uniform(0.1, 0.2)),
                'defensive_efficiency': float(np.random.uniform(0.5, 0.9)),
                'clean_sheet_rate': float(np.random.uniform(0.1, 0.4)),
                'home_strength': float(np.random.uniform(1.0, 1.3)),
                'away_strength': float(np.random.uniform(0.7, 1.0)),
                'points_per_match': float(np.random.uniform(1.0, 2.5)),
                'current_form': float(np.random.uniform(0.2, 0.9)),
                'consistency_score': float(np.random.uniform(0.6, 0.9)),
                'comprehensive_score': float(np.random.uniform(30, 80))
            }
        
        return fallback_data
    
    def run_comprehensive_prediction_pipeline(self, weeks_ahead: int = 2) -> Dict:
        """تشغيل خط أنابيب التنبؤ الشامل"""
        self.logger.info("🔮 بدء خط أنابيب التنبؤ الشامل...")
        
        if not self.prediction_engine:
            self.initialize_all_components()
        
        try:
            # الحصول على المباريات القادمة
            upcoming_matches = self.season_manager.calendar.get_upcoming_matches_with_results(weeks_ahead)
            
            predictions = []
            comparison_data = {
                'timestamp': datetime.now().isoformat(),
                'total_matches': len(upcoming_matches),
                'model_comparison': {},
                'match_predictions': []
            }
            
            for match in upcoming_matches:
                try:
                    home_team = match['home_team']
                    away_team = match['away_team']
                    
                    # توليد تنبؤ شامل
                    comprehensive_pred = self.prediction_engine.generate_comprehensive_prediction(
                        home_team, away_team, "home", None
                    )
                    
                    predictions.append({
                        'match_info': match,
                        'comprehensive_prediction': comprehensive_pred,
                        'generated_at': datetime.now().isoformat()
                    })
                    
                    # جمع بيانات المقارنة
                    match_comparison = {
                        'match': f"{home_team} vs {away_team}",
                        'models': comprehensive_pred.get('model_scores', {}),
                        'consensus': comprehensive_pred.get('consensus_prediction', {}),
                        'betting': comprehensive_pred.get('betting_predictions', {}),
                        'recommendations': comprehensive_pred.get('recommendations', [])
                    }
                    comparison_data['match_predictions'].append(match_comparison)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ فشل تنبؤ للمباراة {match['home_team']} vs {match['away_team']}: {e}")
                    continue
            
            # حفظ التنبؤات
            self.output_manager.save_prediction_batch(predictions, "comprehensive_predictions")
            
            # تحليل أداء النماذج
            model_performance = self._analyze_model_performance(comparison_data)
            comparison_data['model_performance'] = model_performance
            
            # حفظ مقارنة النماذج
            self.output_manager.save_prediction_comparison(comparison_data, "comprehensive_analysis")
            
            # حفظ أداء كل نموذج
            for model_name, performance in model_performance.items():
                self.output_manager.save_prediction_performance(performance, model_name)
            
            self.logger.info(f"✅ اكتمل خط أنابيب التنبؤ: {len(predictions)} تنبؤ")
            
            return {
                'predictions': predictions,
                'comparison': comparison_data,
                'performance': model_performance
            }
            
        except Exception as e:
            self.logger.error(f"❌ فشل خط أنابيب التنبؤ: {e}")
            return {'error': str(e)}
    
    def _analyze_model_performance(self, comparison_data: Dict) -> Dict:
        """تحليل أداء النماذج المختلفة"""
        performance = {
            'advanced': {'total_matches': 0, 'total_score': 0.0, 'average_score': 0.0},
            'realistic': {'total_matches': 0, 'total_score': 0.0, 'average_score': 0.0}
        }
        
        try:
            for match_pred in comparison_data.get('match_predictions', []):
                model_scores = match_pred.get('models', {})
                
                for model_name, score_data in model_scores.items():
                    if model_name in performance and 'score' in score_data:
                        performance[model_name]['total_matches'] += 1
                        performance[model_name]['total_score'] += float(score_data['score'])
            
            # حساب المتوسطات
            for model_name in performance:
                if performance[model_name]['total_matches'] > 0:
                    performance[model_name]['average_score'] = float(
                        performance[model_name]['total_score'] / 
                        performance[model_name]['total_matches']
                    )
            
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحليل الأداء: {e}")
            return performance
    
    def train_with_temporal_integration(self, training_episodes: int = 200):
        """تدريب متكامل زمنياً مع جميع المواسم"""
        self.logger.info("🕰️  بدء التدريب المتكامل زمنياً...")
        
        try:
            # تحميل البيانات متعددة المواسم
            multi_season_data = self._load_multi_season_data()
            
            if len(multi_season_data) < 100:
                self.logger.warning("⚠️  بيانات تدريب محدودة، استخدام البيانات الافتراضية")
                multi_season_data = self._get_comprehensive_training_data()
            
            # تقسيم البيانات مع مراعاة التسلسل الزمني
            train_data, validation_data = self._temporal_split_data(multi_season_data)
            
            self.logger.info(f"📚 بيانات التدريب: {len(train_data)}، التحقق: {len(validation_data)}")
            
            # تدريب مع مراعاة التطور الزمني
            self._temporal_training_pipeline(train_data, validation_data, training_episodes)
            
            self.is_trained = True
            self.logger.info("✅ اكتمل التدريب المتكامل زمنياً!")
            
            # حفظ تقرير التدريب
            self._save_training_analysis()
            
            # تحديث مقاييس الأداء
            self.performance_metrics['last_training'] = datetime.now().isoformat()
            self.performance_metrics['training_samples'] = len(train_data)
            self.performance_metrics['validation_samples'] = len(validation_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ فشل التدريب الزمني: {e}")
            return False
    
    def _load_multi_season_data(self) -> List[Dict]:
        """تحميل بيانات متعددة المواسم"""
        self.logger.info("📥 جاري تحميل بيانات المواسم المتعددة...")
        
        try:
            # حفظ بيانات تحميل الرزنامة
            fixtures = self.season_manager.calendar.all_fixtures
            self.output_manager.save_data_loading_report(fixtures, "calendar", "fixtures")
            
            # تحميل البيانات التاريخية
            historical_data = self.data_enricher.load_and_combine_data()
            
            # حفظ البيانات التاريخية
            source_files = [f"england_E0_{year}.csv" for year in range(2020, 2026)]
            self.output_manager.save_historical_data(historical_data, source_files)
            
            # تحويل البيانات التاريخية إلى تنسيق التدريب
            historical_data_dict = historical_data.to_dict('records')
            historical_data_enriched = self.data_enricher.enrich_match_data(historical_data_dict)
            historical_data_prepared = self.data_enricher.prepare_training_data(historical_data_enriched)
            
            # تحميل بيانات الموسم الحالي
            current_season_data = self._load_current_season_2025_data()
            
            # دمج البيانات مع تحديد الموسم
            combined_data = []
            
            # إضافة البيانات التاريخية
            for record in historical_data_prepared:
                record['season_context'] = {
                    'is_historical': True,
                    'season_weight': 0.7,  # وزن أقل للبيانات القديمة
                    'temporal_adjustment': 0.8
                }
                combined_data.append(record)
            
            # إضافة بيانات الموسم الحالي
            for match_data in current_season_data:
                enriched_match = self.season_manager.validate_and_enrich_match_data(match_data)
                enriched_match['season_context'] = {
                    'is_historical': False,
                    'season_weight': 1.0,  # وزن كامل للموسم الحالي
                    'temporal_adjustment': 1.0,
                    'is_current_season': True
                }
                combined_data.append(enriched_match)
            
            # حفظ البيانات المدمجة
            merge_metadata = {
                'total_historical_matches': len(historical_data_prepared),
                'current_season_matches': len(current_season_data),
                'temporal_integration': True,
                'merge_strategy': 'weighted_temporal'
            }
            self.output_manager.save_merged_training_data(combined_data, merge_metadata)
            
            self.logger.info(f"✅ تم دمج {len(combined_data)} مباراة من مواسم متعددة")
            return combined_data
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل البيانات المتعددة: {e}")
            return []
    
    def _load_current_season_2025_data(self) -> List[Dict]:
        """تحميل بيانات الموسم الحالي 2025/2026"""
        current_data = []
        
        # الحصول على المباريات التي لعبت من الرزنامة
        played_matches = self.season_manager.calendar.get_played_matches()
        
        # إذا لم توجد مباريات ملعوبة، نستخدم البيانات الخام من ملف 2025
        if not played_matches:
            try:
                # تحميل البيانات الخام للموسم الحالي
                current_season_raw = self.data_enricher.get_current_season_2025_data()
                # حفظ تقرير تحميل البيانات الخام
                self.output_manager.save_data_loading_report(current_season_raw, "file", "raw_current_season")
                # تحويلها إلى تنسيق التدريب
                current_season_enriched = self.data_enricher.enrich_match_data(current_season_raw)
                current_data = self.data_enricher.prepare_training_data(current_season_enriched)
            except Exception as e:
                self.logger.error(f"❌ خطأ في تحميل البيانات الخام للموسم الحالي: {e}")
        else:
            # استخدام البيانات من الرزنامة
            for match in played_matches:
                match_data = {
                    'team_metrics': self._get_team_metrics_from_history(match['home_team']),
                    'opponent_metrics': self._get_team_metrics_from_history(match['away_team']),
                    'context': {
                        'home_team': match['home_team'],
                        'away_team': match['away_team'],
                        'match_date': match['match_date'],
                        'week_number': match['week_number'],
                        'is_current_season': True
                    },
                    'actual_result': match['actual_result']
                }
                current_data.append(match_data)
        
        self.logger.info(f"📊 تم تحميل {len(current_data)} مباراة من الموسم الحالي")
        return current_data
    
    def _get_team_metrics_from_history(self, team: str) -> Dict:
        """استخراج مقاييس الفريق من الأداء الحالي"""
        history = self.season_manager.calendar._get_team_season_history(team)
        
        return {
            'points_per_match': float(history.get('points_per_match', 1.0)),
            'win_rate': float(len([f for f in history.get('form', []) if f == 'W']) / max(len(history.get('form', [])), 1)),
            'goal_difference': int(history.get('goal_difference', 0)),
            'goals_per_match': float(history.get('goals_for', 0) / max(history.get('matches_played', 1), 1)),
            'conceded_per_match': float(history.get('goals_against', 0) / max(history.get('matches_played', 1), 1)),
            'current_form': float(len([f for f in history.get('form', []) if f == 'W']) / 5.0),
            'home_advantage': 1.15,  # قيمة افتراضية
            'defensive_efficiency': 0.7,  # قيمة افتراضية
            'motivation_factor': float(self._calculate_current_motivation(team, history))
        }
    
    def _calculate_current_motivation(self, team: str, history: Dict) -> float:
        """حساب الحافز الحالي للفريق"""
        try:
            points = history.get('points', 0)
            matches_played = max(history.get('matches_played', 1), 1)
            avg_points = points / matches_played
            
            # فرق المنافسة على البقاء
            if avg_points < 1.0:
                return 1.3
            # فرق المنافسة على المراكز الأوروبية
            elif avg_points > 1.8:
                return 1.2
            # فرق منتصف الجدول
            else:
                return 1.0
        except:
            return 1.0
    
    def _temporal_split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """تقسيم البيانات مع الحفاظ على التسلسل الزمني"""
        # فصل البيانات التاريخية والحالية
        historical_data = [d for d in data if d.get('season_context', {}).get('is_historical', True)]
        current_data = [d for d in data if not d.get('season_context', {}).get('is_historical', True)]
        
        # تقسيم البيانات التاريخية (80% تدريب، 20% اختبار)
        split_point = int(0.8 * len(historical_data))
        train_data = historical_data[:split_point]
        validation_data = historical_data[split_point:]
        
        # إضافة جميع البيانات الحالية للتدريب (لأنها الأحدث)
        train_data.extend(current_data)
        
        return train_data, validation_data
    
    def _temporal_training_pipeline(self, train_data: List[Dict], validation_data: List[Dict], episodes: int):
        """خطة تدريب متكاملة زمنياً"""
        self.logger.info("🧠 بدء التدريب المتكامل زمنياً...")
        
        # تطبيق أوزان زمنية للبيانات
        weighted_train_data = self._apply_temporal_weights(train_data)
        
        # تدريب النماذج مع مراعاة التسلسل الزمني
        def generation_callback(generation, best_fitness, avg_fitness, population_info, best_weights):
            self.output_manager.save_training_generation(
                generation, best_fitness, avg_fitness, population_info, best_weights
            )
        
        genetic_weights = self.genetic_optimizer.evolve(
            weighted_train_data, 
            self._temporal_fitness_function,
            generations=30,
            callback=generation_callback
        )
        
        # تدريب التعلم المعزز
        def episode_callback(episode, reward, exploration_rate, actions, states):
            self.output_manager.save_rl_episode(
                episode, reward, exploration_rate, actions, states
            )
        
        self.smart_optimizer.train(
            weighted_train_data, 
            episodes=min(episodes, 100),
            callback=episode_callback
        )
        
        rl_weights = self.smart_optimizer.get_optimal_weights()
        
        # دمج الأوزان مع مراعاة التطور الزمني
        self.optimal_weights = self._merge_temporal_weights(
            genetic_weights, rl_weights, train_data
        )
        
        # معايرة باستخدام البيانات الحديثة
        self._temporal_calibration(validation_data)
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'training_samples': len(train_data),
            'validation_samples': len(validation_data),
            'temporal_integration': True,
            'current_season_matches': len([d for d in train_data if d.get('season_context', {}).get('is_current_season', False)])
        })
    
    def _apply_temporal_weights(self, data: List[Dict]) -> List[Dict]:
        """تطبيق أوزان زمنية على بيانات التدريب"""
        weighted_data = []
        
        for match_data in data:
            weighted_match = match_data.copy()
            season_context = match_data.get('season_context', {})
            
            # تطبيق وزن زمني
            temporal_weight = season_context.get('temporal_adjustment', 0.7)
            weighted_match['temporal_weight'] = temporal_weight
            
            weighted_data.append(weighted_match)
        
        return weighted_data
    
    def _temporal_fitness_function(self, weights: Dict, data: List[Dict]) -> float:
        """دالة لياقة مع مراعاة البعد الزمني"""
        total_reward = 0.0
        total_weight = 0.0
        
        for match_data in data[:100]:  # عينة عشوائية
            try:
                temporal_weight = match_data.get('temporal_weight', 0.7)
                
                predictions = self.smart_optimizer.generate_predictions(
                    match_data['team_metrics'],
                    match_data['opponent_metrics'],
                    match_data['context'],
                    weights
                )
                
                if predictions:
                    reward = self.smart_optimizer.calculate_reward(
                        predictions, match_data.get('actual_result', {})
                    )
                    total_reward += reward * temporal_weight
                    total_weight += temporal_weight
                    
            except Exception as e:
                continue
        
        return float(total_reward / total_weight) if total_weight > 0 else 0.0
    
    def _merge_temporal_weights(self, genetic_weights: Dict, rl_weights: Dict, 
                              train_data: List[Dict]) -> Dict[str, float]:
        """دمج الأوزان مع مراعاة التطور الزمني"""
        # حساب نسبة البيانات الحديثة
        recent_data_ratio = len([d for d in train_data if not d.get('season_context', {}).get('is_historical', True)]) / len(train_data)
        
        # زيادة وزن التعلم المعزز للبيانات الحديثة
        rl_weight = 0.5 + (recent_data_ratio * 0.3)
        genetic_weight = 1.0 - rl_weight
        
        merged_weights = {}
        all_keys = set(genetic_weights.keys()) | set(rl_weights.keys())
        
        for key in all_keys:
            genetic_val = genetic_weights.get(key, 0.01)
            rl_val = rl_weights.get(key, 0.01)
            
            merged_weights[key] = float((genetic_val * genetic_weight) + (rl_val * rl_weight))
        
        # تطبيع الأوزان
        total = sum(merged_weights.values())
        return {k: float(v/total) for k, v in merged_weights.items()}
    
    def _temporal_calibration(self, validation_data: List[Dict]):
        """معايرة زمنية باستخدام البيانات الحديثة"""
        try:
            recent_validation = [d for d in validation_data 
                               if not d.get('season_context', {}).get('is_historical', True)]
            
            if len(recent_validation) > 10:
                predictions = []
                actuals = []
                
                for match_data in recent_validation:
                    pred = self.predict_match(
                        match_data['team_metrics'],
                        match_data['opponent_metrics'],
                        match_data['context']
                    )
                    if pred and pred.get('predictions'):
                        predictions.extend(pred['predictions'][:1])
                        actuals.append(match_data.get('actual_result', {}))
                
                if predictions:
                    self.confidence_calibrator.collect_calibration_data(predictions, actuals)
                    self.confidence_calibrator.fit()
                    
        except Exception as e:
            self.logger.warning(f"⚠️  المعايرة الزمنية فشلت: {e}")
    
    def predict_match(self, team_metrics: Dict, opponent_metrics: Dict, 
                     context: Dict) -> Dict:
        """تنبؤ بمباراة مع مراعاة السياق الزمني - نسخة محسنة"""
        try:
            # إثراء البيانات بسياق الموسم الحالي
            enriched_context = self.season_manager.get_current_season_context(
                team_metrics, opponent_metrics, context
            )
            
            # استخدام الأوزان المثلى المدربة زمنياً
            if self.optimal_weights:
                predictions = self.smart_optimizer.generate_predictions(
                    team_metrics, opponent_metrics, enriched_context, self.optimal_weights
                )
                
                if predictions:
                    # تطبيق عوامل التكيف الموسمي
                    seasonal_factors = self._get_seasonal_adjustment_factors(enriched_context)
                    adjusted_predictions = self._apply_seasonal_adjustment(
                        predictions, seasonal_factors
                    )
                    
                    return {
                        'predictions': adjusted_predictions,
                        'seasonal_context': enriched_context,
                        'temporal_adjustment_applied': True
                    }
            
            # التنبؤ الاحتياطي
            return self._get_fallback_prediction(team_metrics, opponent_metrics, enriched_context)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التنبؤ: {e}")
            return self._get_fallback_prediction(team_metrics, opponent_metrics, context)
    
    def _get_seasonal_adjustment_factors(self, context: Dict) -> Dict:
        """الحصول على عوامل التعديل الموسمية"""
        factors = {
            'team_adaptation': float(context.get('adaptation_factor', 1.0)),
            'season_stage': float(context.get('season_stage', 0.5)),
            'current_form_impact': 1.0
        }
        
        # تأثير الأداء الحالي
        team_form = context.get('team_current_season', {}).get('form', [])
        if team_form:
            recent_wins = len([f for f in team_form if f == 'W'])
            factors['current_form_impact'] = 1.0 + (recent_wins * 0.1)
        
        return factors
    
    def _apply_seasonal_adjustment(self, predictions: List[Dict], factors: Dict) -> List[Dict]:
        """تطبيق التعديلات الموسمية على التنبؤات"""
        adjusted_predictions = []
        
        for prediction in predictions:
            adjusted_pred = prediction.copy()
            
            # تعديل الثقة بناءً على عوامل الموسم
            base_confidence = prediction.get('confidence', 0.5)
            seasonal_boost = factors.get('team_adaptation', 1.0) * factors.get('current_form_impact', 1.0)
            
            adjusted_confidence = min(0.95, base_confidence * seasonal_boost)
            adjusted_pred['confidence'] = float(adjusted_confidence)
            adjusted_pred['seasonal_factors'] = factors
            
            adjusted_predictions.append(adjusted_pred)
        
        return adjusted_predictions
    
    def _get_fallback_prediction(self, team_metrics: Dict, opponent_metrics: Dict, 
                               context: Dict) -> Dict:
        """تنبؤ احتياطي مع مراعاة السياق الموسمي"""
        try:
            # استخدام الأداء الحالي من الموسم الحالي
            home_team = context.get('home_team', 'Unknown')
            away_team = context.get('away_team', 'Unknown')
            
            home_history = self.season_manager.calendar._get_team_season_history(home_team)
            away_history = self.season_manager.calendar._get_team_season_history(away_team)
            
            if home_history and away_history:
                home_avg_goals = home_history.get('goals_for', 0) / max(home_history.get('matches_played', 1), 1)
                away_avg_goals = away_history.get('goals_for', 0) / max(away_history.get('matches_played', 1), 1)
                home_conceded = home_history.get('goals_against', 0) / max(home_history.get('matches_played', 1), 1)
                away_conceded = away_history.get('goals_against', 0) / max(away_history.get('matches_played', 1), 1)
                
                # حساب تنبؤ واقعي
                home_goals = max(0, int((home_avg_goals + away_conceded) / 2 * 1.1 + 0.3))
                away_goals = max(0, int((away_avg_goals + home_conceded) / 2 * 0.9 + 0.3))
            else:
                # قيم افتراضية مع تعديل موسمي
                home_strength = team_metrics.get('team_strength', 1.0)
                away_strength = opponent_metrics.get('team_strength', 1.0)
                home_goals = max(1, int(home_strength * 1.2))
                away_goals = max(0, int(away_strength * 0.8))
            
            # حساب الثقة بناءً على جودة البيانات
            confidence = 0.5
            if home_history and away_history and home_history.get('matches_played', 0) > 5:
                confidence = 0.7
            
            return {
                'predictions': [{
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'confidence': float(confidence),
                    'is_fallback': True,
                    'seasonal_context': context
                }]
            }
        except Exception as e:
            self.logger.error(f"❌ خطأ في التنبؤ الاحتياطي: {e}")
            # تنبؤ أساسي جداً
            return {
                'predictions': [{
                    'home_goals': 1,
                    'away_goals': 1,
                    'confidence': 0.3,
                    'is_fallback': True,
                    'error': str(e)
                }]
            }
    
    def _save_training_analysis(self):
        """حفظ تحليل التدريب"""
        try:
            analysis = {
                'training_date': datetime.now().isoformat(),
                'temporal_integration': True,
                'current_season_data_used': len(self._load_current_season_2025_data()),
                'historical_seasons_used': 5,  # افتراضي
                'optimal_weights_keys': list(self.optimal_weights.keys()) if self.optimal_weights else [],
                'training_history': self.training_history
            }
            
            training_file = "output/training/temporal_training_analysis.json"
            self.output_manager.save_json(analysis, training_file)
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ تحليل التدريب: {e}")
    
    def _get_comprehensive_training_data(self) -> List[Dict]:
        """إنشاء بيانات تدريب شاملة عند عدم توفر البيانات الحقيقية"""
        self.logger.info("🔄 إنشاء بيانات تدريب شاملة...")
        
        comprehensive_data = []
        teams = self.season_manager.calendar.teams_2025
        
        for i in range(500):  # 500 مباراة افتراضية
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 5) % len(teams)]
            
            # محاكاة تطور القوة عبر المواسم
            base_season = 2020 + (i % 5)
            season_str = f"{base_season}-{base_season+1}"
            
            home_strength = self.temporal_integrator.get_team_strength_for_season(home_team, season_str)
            away_strength = self.temporal_integrator.get_team_strength_for_season(away_team, season_str)
            
            match_data = {
                'team_metrics': self._generate_metrics_from_strength(home_strength, is_home=True),
                'opponent_metrics': self._generate_metrics_from_strength(away_strength, is_home=False),
                'context': {
                    'home_team': home_team,
                    'away_team': away_team,
                    'season': season_str,
                    'is_historical': base_season < 2024
                },
                'actual_result': {
                    'home_goals': max(0, int(np.random.poisson(home_strength * 1.2))),
                    'away_goals': max(0, int(np.random.poisson(away_strength * 0.8)))
                },
                'season_context': {
                    'is_historical': base_season < 2024,
                    'season_weight': self.temporal_integrator.get_seasonal_adjustment("2024-2025", season_str),
                    'temporal_adjustment': 0.7 if base_season < 2024 else 1.0
                }
            }
            
            comprehensive_data.append(match_data)
        
        return comprehensive_data
    
    def _generate_metrics_from_strength(self, strength: float, is_home: bool) -> Dict:
        """توليد مقاييس من قوة الفريق"""
        home_factor = 1.1 if is_home else 0.9
        
        return {
            'points_per_match': float(strength * 1.2),
            'win_rate': float(min(0.8, strength / 2.5)),
            'goal_difference': int((strength - 1.0) * 20),
            'goals_per_match': float(strength * 1.3 * home_factor),
            'conceded_per_match': float(max(0.5, (2.0 - strength) * 0.8)),
            'current_form': float(min(0.9, strength / 2.2)),
            'home_advantage': 1.15 if is_home else 0.85,
            'defensive_efficiency': float(min(0.95, strength / 2.0)),
            'motivation_factor': float(1.0 + (strength * 0.1))
        }
    
    def evaluate_system_performance(self) -> Dict:
        """تقييم الأداء الشامل للنظام"""
        performance = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'training_status': self.is_trained,
            'component_status': {
                'team_assessor': False,
                'betting_predictor': False,
                'data_validator': False,
                'prediction_engine': bool(self.prediction_engine)
            },
            'data_metrics': {
                'historical_matches': len(self._load_multi_season_data()),
                'current_season_matches': len(self._load_current_season_2025_data()),
                'upcoming_matches': len(self.season_manager.calendar.get_upcoming_matches_with_results())
            },
            'performance_scores': self.performance_metrics
        }
        
        # حساب درجة أداء عامة
        component_score = sum(1 for status in performance['component_status'].values() if status) / len(performance['component_status'])
        data_score = min(1.0, performance['data_metrics']['historical_matches'] / 1000)  # تطبيع
        
        performance['overall_score'] = float((component_score + data_score) / 2)
        
        if performance['overall_score'] >= 0.8:
            performance['system_health'] = 'EXCELLENT'
        elif performance['overall_score'] >= 0.6:
            performance['system_health'] = 'GOOD'
        elif performance['overall_score'] >= 0.4:
            performance['system_health'] = 'FAIR'
        else:
            performance['system_health'] = 'POOR'
        
        return performance

# ==================== التشغيل الرئيسي المحسن بشكل كامل ====================

def main_enhanced():
    # إنشاء هيكل المجلدات أولاً
    output_manager = OutputManager()
    
    system = EnhancedBettingSystem2025("data/football-data")
    
    print("🧠 النظام المحسن للتنبؤ بكرة القدم - الموسم 2025/2026")
    print("🕰️  التكامل الزمني الكامل مع المواسم السابقة")
    print("📊 بيانات حقيقية من football-data.co.uk")
    print("🔮 محرك تنبؤ متكامل مع نماذج متعددة")
    print("🎯 مكونات جديدة: تقييم الفرق، محرك الرهان، التحقق من البيانات")
    print("📁 هيكل الإخراج: output/ مع مجلدات منظمة")
    print("🔄 نظام إعادة التدريب التلقائي")
    print("🔍 التحقق من الدقة الشامل")
    print("🎰 التكامل الكامل لأنواع الرهان")
    print("=" * 60)
    
    try:
        # تدريب متكامل زمنياً
        print("🕰️  بدء التدريب المتكامل زمنياً...")
        training_success = system.train_with_temporal_integration(training_episodes=150)
        
        if training_success:
            print("✅ اكتمل التدريب المتكامل زمنياً!")
            
            # تهيئة جميع المكونات
            print("🔧 تهيئة جميع مكونات النظام...")
            system.initialize_all_components()
            
            # تشغيل خط أنابيب التنبؤ الشامل
            print("🚀 تشغيل خط أنابيب التنبؤ الشامل...")
            prediction_results = system.run_comprehensive_prediction_pipeline(weeks_ahead=3)
            
            # عرض النتائج
            predictions = prediction_results.get('predictions', [])
            comparison = prediction_results.get('comparison', {})
            performance = prediction_results.get('performance', {})
            
            print(f"📊 نتائج التنبؤ الشامل:")
            print(f"• تنبؤات المباريات: {len(predictions)}")
            print(f"• مقارنة النماذج: {len(comparison.get('match_predictions', []))} مباراة")
            print(f"• أداء النماذج: {performance}")
            
            # عرض عينة من التنبؤات
            if predictions:
                print(f"\n🏆 تنبؤات الأسبوع القادم (إجماع النماذج):")
                for i, pred in enumerate(predictions[:5], 1):
                    match_info = pred['match_info']
                    comprehensive = pred['comprehensive_prediction']
                    consensus = comprehensive.get('consensus_prediction', {})
                    betting = comprehensive.get('betting_predictions', {})
                    
                    print(f"{i}. {match_info['home_team']} vs {match_info['away_team']}")
                    print(f"   📅 الأسبوع: {match_info['week_number']} | 📅 التاريخ: {match_info['match_date']}")
                    print(f"   🎯 التنبؤ الإجماعي: {consensus.get('home_goals', 0)}-{consensus.get('away_goals', 0)}")
                    print(f"   💪 الثقة: {consensus.get('confidence', 0.5):.1%}")
                    
                    # عرض تنبؤات الرهان
                    if betting and 'recommended_bets' in betting:
                        top_bet = betting['recommended_bets'][0] if betting['recommended_bets'] else {}
                        if top_bet:
                            print(f"   🎰 أفضل رهان: {top_bet.get('type', '').upper()} - {top_bet.get('selection', '')} (ثقة: {top_bet.get('confidence', 0):.1%})")
                    
                    # عرض توصيات
                    recommendations = comprehensive.get('recommendations', [])
                    if recommendations:
                        print(f"   💡 التوصية: {recommendations[0]}")
                    
                    print()
            
            # تقييم أداء النظام
            system_performance = system.evaluate_system_performance()
            print(f"\n📈 أداء النظام الشامل:")
            print(f"• حالة النظام: {system_performance.get('system_health', 'غير معروف')}")
            print(f"• درجة الأداء: {system_performance.get('overall_score', 0):.1%}")
            print(f"• حالة التدريب: {'مدرب' if system.is_trained else 'غير مدرب'}")
            print(f"• بيانات التدريب: {system_performance['data_metrics']['historical_matches']} مباراة تاريخية")
            print(f"• بيانات الموسم الحالي: {system_performance['data_metrics']['current_season_matches']} مباراة")
            
            return {
                'success': True,
                'system': system,
                'prediction_results': prediction_results,
                'system_performance': system_performance,
                'output_directory': 'output/'
            }
        else:
            print("❌ فشل في تدريب النظام!")
            return {'success': False, 'error': 'Training failed'}
            
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # تشغيل النظام المحسن
    results = main_enhanced()
    
    if results['success']:
        print("🎉 اكتمل النظام المحسن بنجاح!")
        print(f"📁 جميع المخرجات محفوظة في: {results['output_directory']}")
    else:
        print(f"❌ فشل النظام: {results.get('error', 'Unknown error')}")