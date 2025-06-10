from pydantic import BaseModel
from typing import Dict, Any

class ADASData(BaseModel):
    vehicle_speed_kmh: float
    distance_to_obstacle_m: float
    lane_departure_warning: bool
    front_collision_warning: bool
    driver_attention_level: int
    weather_conditions: str
    road_type: str
    light_conditions: str
    steering_angle_degrees: float
    accelerator_pedal_position: float
    brake_pedal_position: float