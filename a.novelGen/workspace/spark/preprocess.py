from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim
import os

def preprocess_data(input_dir, output_dir):
    # Spark 세션 생성
    spark = SparkSession.builder \
        .appName("Novel Preprocessing") \
        .master("spark://anovelgen-spark-master:7077") \
        .getOrCreate()

    # 입력 디렉토리 내 모든 파일 리스트 가져오기
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    # 각 파일별로 처리 (넘버링 적용)
    for idx, file_name in enumerate(file_list, start=1):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"preprocessed_{idx}.txt")

        # 텍스트 파일 읽기
        data = spark.read.text(input_path)

        # 데이터 전처리
        processed_data = (
            data
            .withColumn("value", regexp_replace(col("value"), r"\s+", " "))  # 여러 공백 하나로
            .withColumn("value", trim(col("value")))                        # 앞뒤 공백 제거
            .filter(col("value") != "")                                    # 빈 줄 제거
        )

        # 결과 저장
        processed_data.write.mode("overwrite").text(output_path)
        print(f"Processed {input_path} -> {output_path}")

    spark.stop()

if __name__ == "__main__":
    input_dir = "/data/raw_data/"       # 원본 데이터 디렉토리
    output_dir = "/data/processed_data/"  # 전처리된 데이터 디렉토리
    preprocess_data(input_dir, output_dir)
