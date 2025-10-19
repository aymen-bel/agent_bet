# current_season_config.py - إعدادات وتكوين الموسم الحالي
from datetime import datetime
from typing import Dict, List

class CurrentSeasonConfig:
    """تكوين الموسم الحالي"""
    
    # إعدادات البيانات
    DATA_SOURCES = {
        'primary': 'data/current_season',
        'backup': 'data/archive',
        'live_api': 'https://api.football-data.org/v4'
    }
    
    # إعدادات النماذج
    MODEL_SETTINGS = {
        'training_episodes': 200,
        'genetic_generations': 30,
        'neural_epochs': 100,
        'fast_retrain_episodes': 100
    }
    
    # إعدادات الرهانات
    BETTING_SETTINGS = {
        'initial_capital': 1000.0,
        'max_bet_percentage': 0.02,
        'min_confidence': 0.45,
        'max_daily_bets': 8,
        'daily_loss_limit': 0.1,  # 10%
        'total_loss_limit': 0.2   # 20%
    }
    
    # إعدادات إدارة المخاطر
    RISK_MANAGEMENT = {
        'enable_stop_loss': True,
        'enable_take_profit': True,
        'max_concurrent_bets': 5,
        'cooldown_after_loss': 3,  # أيام
        'min_odds': 1.5,
        'max_odds': 4.0
    }
    
    # الدوري والفرق المفضلة
    FOCUS_LEAGUES = [
        'premier_league',
        'la_liga', 
        'serie_a',
        'bundesliga',
        'ligue_1'
    ]
    
    # الفرق التي يتطلب تحليلها بشكل خاص
    KEY_TEAMS = [
        'Man City', 'Liverpool', 'Arsenal', 'Chelsea',
        'Real Madrid', 'Barcelona', 'Bayern Munich',
        'PSG', 'Juventus', 'AC Milan'
    ]
    
    @classmethod
    def get_current_season_parameters(cls) -> Dict:
        """معاملات الموسم الحالي"""
        current_year = datetime.now().year
        season_start = datetime(current_year, 8, 1)
        season_end = datetime(current_year + 1, 5, 31)
        
        days_elapsed = (datetime.now() - season_start).days
        total_days = (season_end - season_start).days
        season_progress = min(max(days_elapsed / total_days, 0.0), 1.0)
        
        return {
            'season_year': f"{current_year}-{current_year + 1}",
            'season_progress': season_progress,
            'current_stage': cls._get_season_stage(season_progress),
            'transfer_window_active': cls._is_transfer_window_active(),
            'international_break': cls._is_international_break()
        }
    
    @classmethod
    def _get_season_stage(cls, progress: float) -> str:
        """تحديد مرحلة الموسم"""
        if progress < 0.2:
            return "EARLY_SEASON"
        elif progress < 0.4:
            return "MID_SEASON"
        elif progress < 0.7:
            return "LATE_SEASON"
        else:
            return "FINAL_STRETCH"
    
    @classmethod
    def _is_transfer_window_active(cls) -> bool:
        """هل نافذة الانتقالات مفتوحة؟"""
        today = datetime.now()
        # نافذة الصيف: يونيو-أغسطس، الشتاء: يناير
        return (today.month in [6, 7, 8]) or (today.month == 1)
    
    @classmethod
    def _is_international_break(cls) -> bool:
        """هل هناك توقف لبطولات المنتخبات؟"""
        today = datetime.now()
        # أشهر التوقف الدولية التقريبية
        international_months = [3, 6, 9, 10, 11]
        return today.month in international_months
    
    @classmethod
    def get_league_priorities(cls) -> Dict[str, float]:
        """أولويات الدوريات بناءً على مرحلة الموسم"""
        season_params = cls.get_current_season_parameters()
        stage = season_params['current_stage']
        
        if stage == "EARLY_SEASON":
            return {
                'premier_league': 1.0,
                'la_liga': 0.9,
                'bundesliga': 0.8,
                'serie_a': 0.8,
                'ligue_1': 0.7
            }
        elif stage == "FINAL_STRETCH":
            return {
                'premier_league': 1.0,
                'la_liga': 0.95,
                'serie_a': 0.9,
                'bundesliga': 0.85,
                'ligue_1': 0.8
            }
        else:
            return {
                'premier_league': 1.0,
                'la_liga': 0.9,
                'serie_a': 0.85,
                'bundesliga': 0.85,
                'ligue_1': 0.75
            }

# إعدادات التسجيل
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': f'logs/current_season_{datetime.now().strftime("%Y%m%d")}.log',
            'formatter': 'detailed',
            'encoding': 'utf-8'
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
        }
    }
}