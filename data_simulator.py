import time
import random
from datetime import datetime, timedelta
from faker import Faker
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

fake = Faker()

MONGO_URI = "MONGO_URI"
DATABASE_NAME = "mlops_pipeline"
COLLECTION_NAME = "sensor_readings"

FLIP_PROBABILITY = 0.08

def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("MongoDB connection successful!")
        return client
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        return None

def generate_adas_record(record_id: int):
    timestamp = datetime.now() - timedelta(seconds=random.randint(0, 3600))
    
    vehicle_speed = round(random.uniform(0, 120), 2)
    
    if vehicle_speed < 30:
        distance_obstacle = round(random.uniform(0.5, 50), 2)
    elif vehicle_speed < 80:
        distance_obstacle = round(random.uniform(10, 150), 2)
    else:
        distance_obstacle = round(random.uniform(50, 300), 2)

    lane_departure_warning = False
    if vehicle_speed > 60 and random.random() < 0.15:
        lane_departure_warning = True
    elif distance_obstacle < 5 and random.random() < 0.2:
        lane_departure_warning = True

    front_collision_warning = False
    if distance_obstacle < 10 and vehicle_speed > 40 and random.random() < 0.25:
        front_collision_warning = True
    elif distance_obstacle < 2 and random.random() < 0.8:
        front_collision_warning = True

    driver_attention = random.randint(70, 100)
    if lane_departure_warning or front_collision_warning:
        driver_attention = random.randint(30, 70)
    
    weather_conditions = random.choice(['Clear', 'Rainy', 'Foggy', 'Snowy'])
    
    road_type = random.choice(['Highway', 'Urban', 'Rural'])

    light_conditions = random.choice(['Daylight', 'Night', 'Dusk/Dawn'])

    steering_angle = round(random.uniform(-10, 10), 2)
    if lane_departure_warning:
        steering_angle = round(random.uniform(-30, 30), 2) 

    accelerator_pedal = round(random.uniform(0, 1), 2)

    brake_pedal = round(random.uniform(0, 0.2), 2)
    if random.random() < 0.1:
        brake_pedal = round(random.uniform(0.5, 1), 2)

    if front_collision_warning:
        brake_pedal = round(random.uniform(0.5, 1), 2)

    latitude = round(random.uniform(18.9, 19.3), 6)
    longitude = round(random.uniform(72.7, 73.1), 6)

    intervention_needed = False
    if (front_collision_warning or lane_departure_warning or 
        (distance_obstacle < 5 and vehicle_speed > 60) or 
        (driver_attention < 50 and vehicle_speed > 30)):
        intervention_needed = True

    if random.random() < FLIP_PROBABILITY:
        intervention_needed = not intervention_needed

    return {
        "record_id": record_id,
        "timestamp": timestamp,
        "vehicle_speed_kmh": vehicle_speed,
        "distance_to_obstacle_m": distance_obstacle,
        "lane_departure_warning": lane_departure_warning,
        "front_collision_warning": front_collision_warning,
        "driver_attention_level": driver_attention,
        "weather_conditions": weather_conditions,
        "road_type": road_type,
        "light_conditions": light_conditions,
        "steering_angle_degrees": steering_angle,
        "accelerator_pedal_position": accelerator_pedal,
        "brake_pedal_position": brake_pedal,
        "gps_latitude": latitude,
        "gps_longitude": longitude,
        "intervention_needed": intervention_needed
    }

def main():
    client = get_mongo_client()
    if not client:
        return

    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    record_count = 0
    try:
        while True:
            record_id = record_count + 1
            data = generate_adas_record(record_id)
            
            try:
                collection.insert_one(data)
                # print(f"Inserted record {record_id} at {data['timestamp']}: Speed={data['vehicle_speed_kmh']} km/h, Obstacle={data['distance_to_obstacle_m']} m, FCW={data['front_collision_warning']}, LDM={data['lane_departure_warning']}")
                record_count += 1
            except OperationFailure as e:
                print(f"Failed to insert document: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during insertion: {e}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nData generation stopped by user.")
    finally:
        if client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    main()
