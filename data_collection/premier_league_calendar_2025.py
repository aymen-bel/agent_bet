# premier_league_calendar_2025.py
from datetime import datetime, timedelta

class PremierLeagueCalendar2025:
    def __init__(self):
        self.season_start = datetime(2025, 8, 9)  # تاريخ بداية الموسم
        self.season_end = datetime(2026, 5, 24)   # تاريخ نهاية الموسم
        self.teams_2025 = [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
            'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham',
            'West Ham United', 'Wolves'
        ]
        
        # جدول المباريات (مبسط لأغراض العرض)
        self.match_weeks = self._generate_season_calendar()
    
    def _generate_season_calendar(self):
        """توليد رزنامة الموسم (تبديل بين الفرق)"""
        match_weeks = {}
        
        # 38 أسبوعًا في الموسم
        for week in range(1, 39):
            start_date = self.season_start + timedelta(days=(week-1)*7)
            end_date = start_date + timedelta(days=6)
            
            match_weeks[week] = {
                'week_number': week,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'matches': self._generate_week_matches(week)
            }
        
        return match_weeks
    
    def _generate_week_matches(self, week):
        """توليد مباريات الأسبوع (نموذج مبسط)"""
        matches = []
        
        # توليد 10 مباريات لكل أسبوع (20 فريق)
        for i in range(0, 20, 2):
            home_idx = (week + i) % 20
            away_idx = (week + i + 1) % 20
            
            matches.append({
                'home_team': self.teams_2025[home_idx],
                'away_team': self.teams_2025[away_idx],
                'match_date': (self.season_start + timedelta(days=(week-1)*7 + i//2)).strftime('%Y-%m-%d'),
                'venue': f"{self.teams_2025[home_idx]} Stadium"
            })
        
        return matches
    
    def get_current_week(self):
        """الحصول على الأسبوع الحالي من الموسم"""
        today = datetime.now()
        if today < self.season_start:
            return 0  # قبل بداية الموسم
        elif today > self.season_end:
            return 39  # بعد نهاية الموسم
        
        days_elapsed = (today - self.season_start).days
        current_week = (days_elapsed // 7) + 1
        
        return min(current_week, 38)
    
    def get_week_matches(self, week_number):
        """الحصول على مباريات أسبوع محدد"""
        return self.match_weeks.get(week_number, {}).get('matches', [])
    
    def get_upcoming_matches(self, weeks_ahead=2):
        """الحصول على المباريات القادمة"""
        current_week = self.get_current_week()
        upcoming_matches = []
        
        for week in range(current_week, min(current_week + weeks_ahead + 1, 39)):
            week_matches = self.get_week_matches(week)
            for match in week_matches:
                match['week_number'] = week
                upcoming_matches.append(match)
        
        return upcoming_matches
    
    def get_team_fixtures(self, team_name, remaining_only=True):
        """الحصول على مباريات فريق معين"""
        fixtures = []
        
        start_week = self.get_current_week() if remaining_only else 1
        for week in range(start_week, 39):
            week_matches = self.get_week_matches(week)
            for match in week_matches:
                if match['home_team'] == team_name or match['away_team'] == team_name:
                    match['week_number'] = week
                    match['is_home'] = match['home_team'] == team_name
                    fixtures.append(match)
        
        return fixtures