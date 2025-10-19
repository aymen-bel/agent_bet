import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time
from typing import List, Dict, Optional
import logging

class FutureFixturesLoader:
    """محمل الرزنامة المستقبلية المحسن مع التعامل الدقيق مع التوقيت"""
    
    def __init__(self, api_key: str = "a12ac8d7e572423983c7d0ce68513c31"):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': self.api_key}
        self.season_start = datetime(2025, 8, 9)
        self.season_end = datetime(2026, 5, 24)
        self.current_datetime = datetime.now()  # الوقت الحالي الفعلي
        self.output_dir = "output/data_loading"
        os.makedirs(self.output_dir, exist_ok=True)

        # إعداد التسجيل
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_premier_league_fixtures(self, season: int = 2025) -> List[Dict]:
        """تحميل رزنامة الدوري الإنجليزي للموسم 2025/2026"""
        self.logger.info(f"📥 جاري تحميل رزنامة الدوري الإنجليزي موسم {season}...")
        
        try:
            # الحد من معدل الطلبات
            time.sleep(6)
            
            url = f"{self.base_url}/competitions/PL/matches"
            params = {
                'season': season,
                'status': 'SCHEDULED,TIMED,LIVE,FINISHED,POSTPONED'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                fixtures = self.parse_api_response(data)
                self.logger.info(f"✅ تم تحميل {len(fixtures)} مباراة من API")
                return fixtures
            elif response.status_code == 429:
                self.logger.warning("⚠️  تجاوز الحد المسموح من الطلبات، استخدام البيانات المحلية...")
                return self.load_local_fixtures()
            else:
                self.logger.error(f"❌ خطأ في API: {response.status_code}")
                return self.load_local_fixtures()
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل البيانات من API: {e}")
            return self.load_local_fixtures()
    
    def parse_api_response(self, api_data: Dict) -> List[Dict]:
        """تحليل استجابة API وتحويلها إلى تنسيق النظام مع التعامل الدقيق مع الوقت"""
        fixtures = []
        
        for match in api_data.get('matches', []):
            try:
                # استخراج البيانات الأساسية
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                
                # تحويل التاريخ والوقت
                match_datetime = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
                
                # تحديد حالة المباراة بناءً على الوقت الحالي
                status = match['status']
                
                # حساب إذا كانت المباراة مستقبلية (بعد الوقت الحالي)
                is_future = match_datetime > self.current_datetime
                
                # تحديد إذا كانت المباراة لعبت (انتهت قبل الوقت الحالي)
                is_played = status in ['FINISHED', 'AWARDED'] and not is_future
                
                # استخراج النتائج إذا كانت المباراة انتهت
                home_goals = None
                away_goals = None
                if is_played and 'score' in match:
                    home_goals = match['score']['fullTime']['home']
                    away_goals = match['score']['fullTime']['away']
                
                # حساب أسبوع المباراة بدقة
                matchday = self.calculate_exact_matchday(match_datetime)
                
                fixture = {
                    'MatchID': match['id'],
                    'DateTime': match_datetime.isoformat(),  # حفظ كامل التاريخ والوقت
                    'Date': match_datetime.strftime('%d/%m/%Y'),
                    'Time': match_datetime.strftime('%H:%M'),
                    'HomeTeam': self.standardize_team_name(home_team),
                    'AwayTeam': self.standardize_team_name(away_team),
                    'HomeGoals': home_goals,
                    'AwayGoals': away_goals,
                    'Status': status,
                    'Matchday': matchday,
                    'Venue': match.get('venue', {}).get('name', f"{home_team} Stadium"),
                    'Referee': match.get('referee', {}).get('name', 'Unknown'),
                    'IsPlayed': is_played,
                    'IsFuture': is_future,
                    'TimeUntilMatch': self.calculate_time_until_match(match_datetime),
                    'Season': '2025-2026',
                    'API_Data': True,
                    'Competition': match.get('competition', {}).get('name', 'Premier League')
                }
                
                fixtures.append(fixture)
                
            except Exception as e:
                self.logger.warning(f"⚠️  خطأ في معالجة مباراة: {e}")
                continue
        
        return fixtures
    
    def calculate_exact_matchday(self, match_datetime: datetime) -> int:
        """حساب أسبوع المباراة بدقة بناءً على تاريخ بداية الموسم"""
        days_from_start = (match_datetime - self.season_start).days
        matchday = max(1, (days_from_start // 7) + 1)
        return min(matchday, 38)  # أقصى عدد أسابيع في الدوري
    
    def calculate_time_until_match(self, match_datetime: datetime) -> str:
        """حساب الوقت المتبقي حتى المباراة"""
        if match_datetime <= self.current_datetime:
            return "مباراة ماضية"
        
        time_diff = match_datetime - self.current_datetime
        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60
        
        if days > 0:
            return f"بعد {days} يوم و {hours} ساعة"
        elif hours > 0:
            return f"بعد {hours} ساعة و {minutes} دقيقة"
        else:
            return f"بعد {minutes} دقيقة"
    
    def standardize_team_name(self, team_name: str) -> str:
        """توحيد أسماء الفرق للتطابق مع النظام"""
        name_mapping = {
            'Liverpool FC': 'Liverpool',
            'AFC Bournemouth': 'Bournemouth',
            'Aston Villa FC': 'Aston Villa',
            'Newcastle United FC': 'Newcastle United',
            'Brighton & Hove Albion FC': 'Brighton',
            'Fulham FC': 'Fulham',
            'Sunderland AFC': 'Sunderland',
            'West Ham United FC': 'West Ham United',
            'Tottenham Hotspur FC': 'Tottenham',
            'Burnley FC': 'Burnley',
            'Wolverhampton Wanderers FC': 'Wolverhampton Wanderers',
            'Manchester City FC': 'Manchester City',
            'Manchester United FC': 'Manchester United',
            'Chelsea FC': 'Chelsea',
            'Arsenal FC': 'Arsenal',
            'Crystal Palace FC': 'Crystal Palace',
            'Everton FC': 'Everton',
            'Leicester City FC': 'Leicester City',
            'Southampton FC': 'Southampton',
            'Ipswich Town FC': 'Ipswich Town',
            'Nottingham Forest FC': 'Nottingham Forest',
            'Brentford FC': 'Brentford'
        }
        return name_mapping.get(team_name, team_name)
    
    def load_local_fixtures(self) -> List[Dict]:
        """تحميل البيانات المحلية إذا فشل API"""
        self.logger.info("🔄 استخدام البيانات المحلية كبديل...")
        
        try:
            # محاولة تحميل البيانات المحفوظة مسبقاً
            local_files = [
                "data/seasons/premier_league_2025_fixtures.json",
                "data/seasons/england_E0_2025.csv",
                "data/seasons/premier_league_2025_fixtures.csv"
            ]
            
            for local_file in local_files:
                if os.path.exists(local_file):
                    if local_file.endswith('.json'):
                        with open(local_file, 'r', encoding='utf-8') as f:
                            fixtures = json.load(f)
                        self.logger.info(f"✅ تم تحميل {len(fixtures)} مباراة من الملف المحلي JSON")
                        return self.enhance_local_fixtures(fixtures)
                    elif local_file.endswith('.csv'):
                        df = pd.read_csv(local_file)
                        fixtures = self.convert_csv_to_fixtures(df)
                        self.logger.info(f"✅ تم تحميل {len(fixtures)} مباراة من الملف المحلي CSV")
                        return fixtures
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل الملفات المحلية: {e}")
        
        # إنشاء بيانات محاكاة كحل أخير
        return self.generate_realistic_fixtures()
    
    def enhance_local_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """تحسين البيانات المحلية بإضافة المعلومات الزمنية المفقودة"""
        enhanced_fixtures = []
        
        for fixture in fixtures:
            try:
                # تحليل التاريخ والوقت من البيانات المحلية
                date_str = fixture.get('Date', '')
                time_str = fixture.get('Time', '15:00')
                
                # معالجة تنسيقات التاريخ المختلفة
                match_datetime = self.parse_flexible_datetime(date_str, time_str)
                
                if match_datetime:
                    # تحديث الحقول الزمنية
                    fixture['DateTime'] = match_datetime.isoformat()
                    fixture['IsFuture'] = match_datetime > self.current_datetime
                    fixture['IsPlayed'] = fixture.get('IsPlayed', False) and not fixture['IsFuture']
                    fixture['TimeUntilMatch'] = self.calculate_time_until_match(match_datetime)
                    
                    # تحديث أسماء الفرق
                    fixture['HomeTeam'] = self.standardize_team_name(fixture.get('HomeTeam', ''))
                    fixture['AwayTeam'] = self.standardize_team_name(fixture.get('AwayTeam', ''))
                
                enhanced_fixtures.append(fixture)
                
            except Exception as e:
                self.logger.warning(f"⚠️  خطأ في تحسين المباراة المحلية: {e}")
                continue
        
        return enhanced_fixtures
    
    def parse_flexible_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        """تحليل تنسيقات التاريخ والوقت المختلفة"""
        try:
            # تنسيق DD/MM/YYYY
            if '/' in date_str:
                date_part = datetime.strptime(date_str, '%d/%m/%Y')
            # تنسيق YYYY-MM-DD
            elif '-' in date_str:
                date_part = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                # استخدام تاريخ افتراضي إذا فشل التحليل
                date_part = self.season_start + timedelta(days=len(date_str) * 7)
            
            # تحليل الوقت
            if ':' in time_str:
                time_part = datetime.strptime(time_str, '%H:%M').time()
            else:
                time_part = datetime.strptime('15:00', '%H:%M').time()  # وقت افتراضي
            
            return datetime.combine(date_part.date(), time_part)
            
        except Exception:
            # استخدام تاريخ افتراضي
            default_date = self.season_start + timedelta(weeks=len(date_str))
            return datetime.combine(default_date.date(), datetime.strptime('15:00', '%H:%M').time())
    
    def convert_csv_to_fixtures(self, df: pd.DataFrame) -> List[Dict]:
        """تحويل بيانات CSV إلى تنسيق الفيكستشر"""
        fixtures = []
        
        for _, row in df.iterrows():
            try:
                # معالجة التاريخ والوقت
                date_str = str(row.get('Date', ''))
                time_str = '15:00'  # وقت افتراضي
                
                match_datetime = self.parse_flexible_datetime(date_str, time_str)
                is_played = pd.notna(row.get('FTHG', None))
                
                fixture = {
                    'MatchID': hash(f"{row['HomeTeam']}_{row['AwayTeam']}_{date_str}"),
                    'DateTime': match_datetime.isoformat() if match_datetime else '',
                    'Date': match_datetime.strftime('%d/%m/%Y') if match_datetime else date_str,
                    'Time': match_datetime.strftime('%H:%M') if match_datetime else time_str,
                    'HomeTeam': self.standardize_team_name(row['HomeTeam']),
                    'AwayTeam': self.standardize_team_name(row['AwayTeam']),
                    'HomeGoals': row.get('FTHG', None) if is_played else None,
                    'AwayGoals': row.get('FTAG', None) if is_played else None,
                    'Status': 'FINISHED' if is_played else 'SCHEDULED',
                    'Matchday': self.calculate_exact_matchday(match_datetime) if match_datetime else 1,
                    'Venue': f"{row['HomeTeam']} Stadium",
                    'Referee': 'Unknown',
                    'IsPlayed': is_played,
                    'IsFuture': match_datetime > self.current_datetime if match_datetime else False,
                    'TimeUntilMatch': self.calculate_time_until_match(match_datetime) if match_datetime else "غير معروف",
                    'Season': '2025-2026',
                    'API_Data': False
                }
                fixtures.append(fixture)
                
            except Exception as e:
                self.logger.warning(f"⚠️  خطأ في تحويل صف CSV: {e}")
                continue
        
        return fixtures
    
    def generate_realistic_fixtures(self) -> List[Dict]:
        """إنشاء رزنامة محاكاة واقعية مع تواريخ دقيقة"""
        self.logger.info("🔄 إنشاء رزنامة محاكاة واقعية...")
        
        teams = self.get_2025_teams()
        fixtures = []
        match_id = 1000000
        
        # إنشاء جدول المباريات المتوازن
        match_schedule = self.create_balanced_fixture_schedule(teams)
        
        current_date = self.season_start
        week_number = 1
        
        for week_fixtures in match_schedule:
            # توزيع المباريات على أيام الأسبوع بشكل واقعي
            match_dates = self.generate_week_match_dates(current_date, len(week_fixtures))
            
            for i, match in enumerate(week_fixtures):
                if i < len(match_dates):
                    match_datetime = match_dates[i]
                else:
                    match_datetime = current_date + timedelta(days=6, hours=15)  # تاريخ افتراضي
                
                fixture = {
                    'MatchID': match_id,
                    'DateTime': match_datetime.isoformat(),
                    'Date': match_datetime.strftime('%d/%m/%Y'),
                    'Time': match_datetime.strftime('%H:%M'),
                    'HomeTeam': match['home'],
                    'AwayTeam': match['away'],
                    'HomeGoals': None,
                    'AwayGoals': None,
                    'Status': 'SCHEDULED',
                    'Matchday': week_number,
                    'Venue': f"{match['home']} Stadium",
                    'Referee': 'Simulated',
                    'IsPlayed': match_datetime < self.current_datetime,
                    'IsFuture': match_datetime > self.current_datetime,
                    'TimeUntilMatch': self.calculate_time_until_match(match_datetime),
                    'Season': '2025-2026',
                    'API_Data': False
                }
                
                fixtures.append(fixture)
                match_id += 1
            
            current_date += timedelta(days=7)
            week_number += 1
        
        return fixtures
    
    def generate_week_match_dates(self, week_start: datetime, num_matches: int) -> List[datetime]:
        """توليد تواريخ واقعية لمباريات الأسبوع"""
        match_dates = []
        
        # توزيع المباريات على عطلات نهاية الأسبوع بشكل أساسي
        weekend_dates = []
        weekday_dates = []
        
        for day in range(7):
            current_date = week_start + timedelta(days=day)
            if current_date.weekday() in [4, 5, 6]:  # الجمعة، السبت، الأحد
                weekend_dates.extend(self.generate_day_match_times(current_date))
            else:
                weekday_dates.extend(self.generate_day_match_times(current_date))
        
        # إعطاء الأولوية لعطلات نهاية الأسبوع
        all_dates = weekend_dates + weekday_dates
        return all_dates[:num_matches]
    
    def generate_day_match_times(self, date: datetime) -> List[datetime]:
        """توليد أوقات واقعية للمباريات في يوم معين"""
        times = []
        weekday = date.weekday()
        
        if weekday == 5:  # السبت
            times = ['12:30', '15:00', '17:30', '20:00']
        elif weekday == 6:  # الأحد
            times = ['14:00', '16:30', '19:00']
        elif weekday == 4:  # الجمعة
            times = ['20:00', '20:15']
        else:  # أيام الأسبوع
            times = ['19:45', '20:00', '20:15']
        
        return [datetime.combine(date.date(), datetime.strptime(t, '%H:%M').time()) for t in times]
    
    def create_balanced_fixture_schedule(self, teams: List[str]) -> List[List[Dict]]:
        """إنشاء جدول مباريات متوازن (نظام الدوري الإنجليزي)"""
        import random
        random.seed(42)  # للحصول على نتائج متسقة
        
        num_teams = len(teams)
        half_season = num_teams - 1
        all_fixtures = []
        
        # خلط الفرق عشوائياً
        shuffled_teams = teams.copy()
        random.shuffle(shuffled_teams)
        
        # خوارزمية جدولة الدوري
        for round_num in range(half_season * 2):
            round_fixtures = []
            
            for i in range(num_teams // 2):
                home_idx = i
                away_idx = num_teams - 1 - i
                
                # في النصف الثاني من الموسم، نعكس المباريات
                if round_num >= half_season:
                    home_idx, away_idx = away_idx, home_idx
                
                # تجنب الفريق "الراحة" إذا كان عدد الفرق فردياً
                if home_idx < len(shuffled_teams) and away_idx < len(shuffled_teams):
                    round_fixtures.append({
                        'home': shuffled_teams[home_idx], 
                        'away': shuffled_teams[away_idx]
                    })
            
            all_fixtures.append(round_fixtures)
            
            # تدوير الفرق للحصول على جدول متوازن
            shuffled_teams.insert(1, shuffled_teams.pop())
        
        return all_fixtures
    
    def get_2025_teams(self) -> List[str]:
        """قائمة فرق الدوري الإنجليزي 2025/2026"""
        return [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
            'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham',
            'West Ham United', 'Wolverhampton Wanderers'
        ]
    
    def save_fixtures_to_file(self, fixtures: List[Dict], 
                            filename: str = "data/seasons/premier_league_2025_fixtures.json"):
        """حفظ الرزنامة في ملف للاستخدام المستقبلي"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(fixtures, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 تم حفظ {len(fixtures)} مباراة في {filename}")
            
            # حفظ نسخة CSV أيضاً
            csv_filename = filename.replace('.json', '.csv')
            df = pd.DataFrame(fixtures)
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            self.logger.info(f"💾 تم حفظ نسخة CSV في {csv_filename}")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ الملف: {e}")
            return None


    def save_loading_report(self, fixtures: List[Dict], source: str):
        """حفظ تقرير تحميل البيانات"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # حفظ كملف JSON مفصل
        report_data = {
            'load_timestamp': datetime.now().isoformat(),
            'fixtures_count': len(fixtures),
            'data_source': source,
            'future_matches': len([f for f in fixtures if f.get('IsFuture', False)]),
            'played_matches': len([f for f in fixtures if f.get('IsPlayed', False)]),
            'fixtures': fixtures
        }
        
        json_filename = f"{self.output_dir}/fixtures_load_report_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # حفظ كملف CSV مختصر
        csv_data = []
        for fixture in fixtures:
            csv_data.append({
                'MatchID': fixture.get('MatchID'),
                'Date': fixture.get('Date'),
                'HomeTeam': fixture.get('HomeTeam'),
                'AwayTeam': fixture.get('AwayTeam'),
                'Status': fixture.get('Status'),
                'IsFuture': fixture.get('IsFuture', False),
                'IsPlayed': fixture.get('IsPlayed', False)
            })
        
        df = pd.DataFrame(csv_data)
        csv_filename = f"{self.output_dir}/fixtures_summary_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        print(f"💾 تم حفظ تقرير التحميل: {json_filename}")
        print(f"💾 تم حفظ الملخص: {csv_filename}")
        
        return json_filename, csv_filename
    
    def load_fixtures(self, use_cache: bool = True) -> List[Dict]:
        cache_file = "data/seasons/premier_league_2025_fixtures.json"
        
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    fixtures = json.load(f)
                print(f"✅ تم تحميل {len(fixtures)} مباراة من الذاكرة المؤقتة")
                fixtures = self.enhance_local_fixtures(fixtures)
                self.save_loading_report(fixtures, "CACHE")
                return fixtures
            except Exception as e:
                print(f"❌ خطأ في تحميل الذاكرة المؤقتة: {e}")
        
        # تحميل من API
        fixtures = self.load_premier_league_fixtures(2025)
        
        if fixtures:
            self.save_fixtures_to_file(fixtures, cache_file)
            self.save_loading_report(fixtures, "API")
        
        return fixtures
    
    def get_upcoming_fixtures(self, days_ahead: int = 7) -> List[Dict]:
        """الحصول على المباريات القادمة فقط في الأيام القادمة"""
        all_fixtures = self.load_fixtures()
        upcoming_fixtures = []
        
        cutoff_date = self.current_datetime + timedelta(days=days_ahead)
        
        for fixture in all_fixtures:
            try:
                fixture_datetime = datetime.fromisoformat(fixture['DateTime'])
                if self.current_datetime < fixture_datetime <= cutoff_date:
                    upcoming_fixtures.append(fixture)
            except:
                continue
        
        # ترتيب المباريات حسب التاريخ
        upcoming_fixtures.sort(key=lambda x: x.get('DateTime', ''))
        return upcoming_fixtures
    
    def get_complete_calendar(self) -> Dict:
        """الحصول على الرزنامة الكاملة منظمة حسب الأسابيع"""
        all_fixtures = self.load_fixtures()
        calendar = {}
        
        for fixture in all_fixtures:
            matchday = fixture.get('Matchday', 1)
            
            if matchday not in calendar:
                calendar[matchday] = {
                    'week_number': matchday,
                    'start_date': self.calculate_week_start_date(matchday),
                    'matches': []
                }
            
            calendar[matchday]['matches'].append(fixture)
        
        return calendar
    
    def calculate_week_start_date(self, matchday: int) -> str:
        """حساب تاريخ بداية الأسبوع"""
        start_date = self.season_start + timedelta(weeks=matchday-1)
        return start_date.strftime('%Y-%m-%d')

# دالة مساعدة للاستخدام المباشر
def load_premier_league_fixtures():
    """تحميل رزنامة الدوري الإنجليزي"""
    loader = FutureFixturesLoader()
    return loader.load_fixtures()

def get_upcoming_matches(days: int = 7):
    """الحصول على المباريات القادمة فقط"""
    loader = FutureFixturesLoader()
    return loader.get_upcoming_fixtures(days)

if __name__ == "__main__":
    # اختبار الوحدة
    print(f"🕐 الوقت الحالي: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    fixtures = load_premier_league_fixtures()
    print(f"إجمالي المباريات المحملة: {len(fixtures)}")
    
    # تصنيف المباريات
    future_matches = [f for f in fixtures if f.get('IsFuture', False)]
    past_matches = [f for f in fixtures if f.get('IsPlayed', False)]
    upcoming_matches = get_upcoming_matches(7)
    
    print(f"📊 الإحصائيات:")
    print(f"   🏟️  المباريات المستقبلية: {len(future_matches)}")
    print(f"   ✅ المباريات المكتملة: {len(past_matches)}")
    print(f"   🔜 المباريات القادمة في الأسبوع: {len(upcoming_matches)}")
    
    # عرض المباريات القادمة فقط
    print(f"\n🏆 المباريات القادمة في الأسبوع:")
    for i, fixture in enumerate(upcoming_matches[:10]):
        status_icon = "🟢" if fixture.get('IsFuture') else "🔴"
        print(f"{status_icon} {i+1}. {fixture['HomeTeam']} vs {fixture['AwayTeam']}")
        print(f"   📅 {fixture['Date']} ⏰ {fixture['Time']} - الأسبوع {fixture.get('Matchday', '?')}")
        print(f"   🏟️  {fixture['Venue']} - ⏳ {fixture.get('TimeUntilMatch', 'غير معروف')}")
        print()
    
    # عرض الرزنامة الكاملة إذا طلب المستخدم
    if len(upcoming_matches) == 0:
        print("\n📅 عينة من الرزنامة الكاملة:")
        for i, fixture in enumerate(fixtures[:5]):
            status = "مستقبلية" if fixture.get('IsFuture') else "مكتملة"
            print(f"{i+1}. {fixture['HomeTeam']} vs {fixture['AwayTeam']} - {fixture['Date']} ({status})")