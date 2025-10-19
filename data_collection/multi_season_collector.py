# data_collection/multi_season_collector.py
import os
import pandas as pd
import requests
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)

class MultiSeasonDataCollector:
    def __init__(self, base_directory="data/football-data"):
        self.base_directory = base_directory
        os.makedirs(base_directory, exist_ok=True)
        
    def download_season_data(self, league, season, country):
        """تنزيل بيانات موسم معين"""
        season_str = f"{str(season-1)[-2:]}{str(season)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{season_str}/{league}.csv"
        
        filename = f"{country}_{league}_{season}.csv"
        filepath = os.path.join(self.base_directory, filename)
        
        try:
            logging.info(f"جاري تنزيل بيانات موسم {season}...")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logging.info(f"✅ تم حفظ بيانات موسم {season} في {filepath}")
                return True
            else:
                logging.warning(f"⚠️ لم يتم العثور على بيانات لموسم {season}")
                return False
                
        except Exception as e:
            logging.error(f"❌ خطأ في تنزيل موسم {season}: {e}")
            return False
    
    def collect_multiple_seasons(self, league, country, start_season, end_season):
        """جمع بيانات لعدة مواسم"""
        collected_seasons = []
        
        for season in range(start_season, end_season + 1):
            success = self.download_season_data(league, season, country)
            if success:
                collected_seasons.append(season)
            time.sleep(1)  # تجنب الحظر من السيرفر
        
        logging.info(f"تم جمع بيانات {len(collected_seasons)} موسم: {collected_seasons}")
        return collected_seasons
    
    def create_combined_dataset(self, output_file="combined_seasons_data.csv"):
        """إنشاء مجموعة بيانات موحدة من جميع المواسم"""
        all_data = []
        
        for filename in os.listdir(self.base_directory):
            if filename.endswith('.csv') and filename != output_file:
                filepath = os.path.join(self.base_directory, filename)
                try:
                    # استخراج معلومات الموسم من اسم الملف
                    parts = filename.replace('.csv', '').split('_')
                    if len(parts) >= 3:
                        country = parts[0]
                        league = parts[1]
                        season = int(parts[2])
                        
                        df = pd.read_csv(filepath)
                        df['Season'] = season
                        df['Country'] = country
                        df['League'] = league
                        
                        all_data.append(df)
                        logging.info(f"تم تحميل بيانات {filename}")
                        
                except Exception as e:
                    logging.error(f"خطأ في تحميل {filename}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            output_path = os.path.join(self.base_directory, output_file)
            combined_df.to_csv(output_path, index=False)
            logging.info(f"✅ تم إنشاء مجموعة البيانات الموحدة في {output_path}")
            return combined_df
        else:
            logging.warning("❌ لم يتم العثور على بيانات لدمجها")
            return None

# استخدام النظام
if __name__ == "__main__":
    collector = MultiSeasonDataCollector()
    
    # جمع بيانات 5 مواسم سابقة للدوري الإنجليزي
    seasons = collector.collect_multiple_seasons(
        league="E0",  # الدوري الإنجليزي الممتاز
        country="england",
        start_season=2020,
        end_season=2024
    )
    
    # إنشاء مجموعة البيانات الموحدة
    combined_data = collector.create_combined_dataset()