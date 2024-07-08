import pymssql
from faker import Faker
import random
from datetime import datetime, timedelta
from config import Config


fake = Faker('en_US')
fake1 = Faker('ja_JP')

conn = pymssql.connect(Config.SQL_SERVER, Config.SQL_USER, Config.SQL_PASSWORD, Config.SQL_DATABASE)
cursor = conn.cursor()

cursor.execute('''
    IF OBJECT_ID('dbo.Runners', 'U') IS NOT NULL
    DROP TABLE dbo.Runners
    CREATE TABLE dbo.Runners (
        BIB NVARCHAR(50) PRIMARY KEY,
        Name NVARCHAR(100),
        DOB DATE,
        Age INT,
        PhoneNumber NVARCHAR(50),
        Gender NVARCHAR(10),
        Email NVARCHAR(100),
        [National] NVARCHAR(50),  -- Enclose National in square brackets
        DateRegister DATE,
        DistanceType NVARCHAR(20),
        Time NVARCHAR(20),
        Pace NVARCHAR(10),
        Complete NVARCHAR(3) DEFAULT 'YES',
        FinisherReceive NVARCHAR(3) DEFAULT 'NO',
        Images NVARCHAR(MAX),
        Notes NVARCHAR(MAX)
    )
    CREATE INDEX idx_bib ON dbo.Runners (BIB)
''')
conn.commit()


# Function to generate and insert data for a given BIB range and distance type
def generate_data(start_bib, end_bib, distance_type, time_range, pace_distance):
    for bib in range(start_bib, end_bib + 1):
        bib_str = str(bib)
        name = fake.name()
        gender = random.choice(['Male', 'Female'])
        email = fake.email()
        dob = fake.date_of_birth(minimum_age=17, maximum_age=60)
        age = 2024 - dob.year
        phone_number = fake1.phone_number()
        national = 'Vietnam'
        date_register = fake.date_between_dates(date_start=datetime(2024, 1, 1), date_end=datetime(2024, 3, 31))

        min_time, max_time = time_range
        total_seconds = random.randint(int(min_time.total_seconds()), int(max_time.total_seconds()))
        time = str(timedelta(seconds=total_seconds))

        pace_minutes = random.randint(4, 12)
        pace_seconds = random.randint(0, 59)
        pace = f"{pace_minutes}:{pace_seconds:02d}"  # M:SS format

        complete = 'YES'
        finisher_receive = 'NO'


        # Insert data into the table
        cursor.execute('''
            INSERT INTO dbo.Runners1 (BIB, Name, DOB, Age, PhoneNumber, Gender, Email, [National], DateRegister, DistanceType, Time, Pace, Complete, FinisherReceive)
            VALUES (%s, %s, %s, %d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
        bib_str, name, dob, age, phone_number, gender, email, national, date_register, distance_type, time, pace, complete,
        finisher_receive))


# Define BIB ranges and corresponding distance types and time ranges
bib_ranges = [
    (50001, 51802, '5km', (timedelta(minutes=20), timedelta(hours=3))),
    (59001, 59010, '5km', (timedelta(minutes=20), timedelta(hours=3))),
    (60001, 62360, '10km', (timedelta(minutes=40), timedelta(hours=4.5))),
    (69001, 69005, '10km', (timedelta(minutes=40), timedelta(hours=4.5))),
    (80001, 83422, '21km', (timedelta(hours=1, minutes=15), timedelta(hours=6))),
    (90001, 90877, '42km', (timedelta(hours=2), timedelta(hours=7))),
    (10001, 10628, 'Kun', (timedelta(minutes=15), timedelta(minutes=40))),
    (3000, 3400, 'Cosplay', (timedelta(minutes=5), timedelta(hours=4))),  # Time is random
]

# Generate data for each range
for start_bib, end_bib, distance_type, time_range in bib_ranges:
    generate_data(start_bib, end_bib, distance_type, time_range, distance_type)


# Commit the transaction
conn.commit()

# Close the connection
conn.close()

print("Data generation completed successfully.")