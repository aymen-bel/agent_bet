# download_seasons.py
import pandas as pd
import requests
import os
from datetime import datetime
import time

class SeasonDataDownloader:
    def __init__(self):
        self.base_url = "https://www.football-data.co.uk/mmz4281"
        self.seasons = {
            '2025': '2526',
            '2024': '2425', 
            '2023': '2324',
            '2022': '2223',
            '2021': '2122',
            '2020': '2021'
        }
        
    def download_all_seasons(self):
        """تحميل جميع مواسم الدوري الإنجليزي"""
        print("📥 بدء تحميل بيانات المواسم من 2020 إلى 2025...")
        
        for year, code in self.seasons.items():
            try:
                self.download_season(year, code)
                time.sleep(1)  # تجنب حظر IP
            except Exception as e:
                print(f"❌ خطأ في تحميل موسم {year}: {e}")
                
        print("✅ اكتمل تحميل جميع المواسم!")
    
    def download_season(self, year, season_code):
        """تحميل موسم محدد"""
        filename = f"england_E0_{year}.csv"
        url = f"{self.base_url}/{season_code}/E0.csv"
        
        try:
            # تحميل البيانات
            response = requests.get(url)
            response.raise_for_status()
            
            # حفظ الملف
            filepath = f"data/seasons/{filename}"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ تم تحميل {filename} بنجاح")
            
            # إنشاء رزنامة الموسم
            self.create_season_calendar(year, filepath)
            
        except Exception as e:
            print(f"❌ فشل تحميل {filename}: {e}")
    
    def create_season_calendar(self, year, data_file):
        """إنشاء رزنامة الموسم من البيانات"""
        try:
            df = pd.read_csv(data_file)
            
            # تنظيف البيانات وإنشاء الرزنامة
            calendar_df = self.clean_fixture_data(df, year)
            
            # حفظ الرزنامة
            calendar_file = f"data/calendar/england_E0_{year}_fixtures.csv"
            calendar_df.to_csv(calendar_file, index=False)
            
            print(f"📅 تم إنشاء رزنامة موسم {year} ({len(calendar_df)} مباراة)")
            
        except Exception as e:
            print(f"❌ خطأ في إنشاء رزنامة {year}: {e}")
    
    def clean_fixture_data(self, df, year):
        """تنظيف بيانات المباريات"""
        # تحديد الأعمدة المطلوبة
        required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        
        # التحقق من وجود الأعمدة
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) < 3:
            raise ValueError(f"بيانات غير كافية في موسم {year}")
        
        # إنشاء DataFrame نظيف
        clean_df = df[available_columns].copy()
        
        # تنظيف التواريخ
        clean_df['Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True, errors='coerce')
        clean_df = clean_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        
        # إضافة معلومات الموسم
        clean_df['Season'] = year
        clean_df['MatchId'] = clean_df['HomeTeam'] + '_' + clean_df['AwayTeam'] + '_' + year
        
        return clean_df

if __name__ == "__main__":
    downloader = SeasonDataDownloader()
    downloader.download_all_seasons()