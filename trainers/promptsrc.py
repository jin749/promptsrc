import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
from collections import defaultdict
from tqdm import tqdm
from dassl.utils import (
    MetricMeter, AverageMeter, load_checkpoint, load_pretrained_weights
)
import time
import datetime

_tokenizer = _Tokenizer()
'''
DescribableTextures(24): -> 50
['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled']
FGVCAircraft(50): -> 100
['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525']
EuroSAT(5): -> 10
['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road', 'Industrial Buildings']
StanfordCars(98): -> 200
['2000 AM General Hummer SUV', '2012 Acura RL Sedan', '2012 Acura TL Sedan', '2008 Acura TL Type-S', '2012 Acura TSX Sedan', '2001 Acura Integra Type R', '2012 Acura ZDX Hatchback', '2012 Aston Martin V8 Vantage Convertible', '2012 Aston Martin V8 Vantage Coupe', '2012 Aston Martin Virage Convertible', '2012 Aston Martin Virage Coupe', '2008 Audi RS 4 Convertible', '2012 Audi A5 Coupe', '2012 Audi TTS Coupe', '2012 Audi R8 Coupe', '1994 Audi V8 Sedan', '1994 Audi 100 Sedan', '1994 Audi 100 Wagon', '2011 Audi TT Hatchback', '2011 Audi S6 Sedan', '2012 Audi S5 Convertible', '2012 Audi S5 Coupe', '2012 Audi S4 Sedan', '2007 Audi S4 Sedan', '2012 Audi TT RS Coupe', '2012 BMW ActiveHybrid 5 Sedan', '2012 BMW 1 Series Convertible', '2012 BMW 1 Series Coupe', '2012 BMW 3 Series Sedan', '2012 BMW3 Series Wagon', '2007 BMW 6 Series Convertible', '2007 BMW X5 SUV', '2012 BMW X6 SUV', '2012 BMW M3 Coupe', '2010 BMW M5 Sedan', '2010 BMW M6 Convertible', '2012 BMW X3 SUV', '2012 BMW Z4 Convertible', '2012 Bentley Continental Supersports Conv. Convertible', '2009 Bentley Arnage Sedan', '2011 Bentley Mulsanne Sedan', '2012 Bentley Continental GT Coupe', '2007 Bentley Continental GT Coupe', '2007 Bentley Continental Flying Spur Sedan', '2009 Bugatti Veyron 16.4 Convertible', '2009 Bugatti Veyron 16.4 Coupe', '2012 Buick Regal GS', '2007 Buick Rainier SUV', '2012 Buick Verano Sedan', '2012 Buick Enclave SUV', '2012 Cadillac CTS-V Sedan', '2012 Cadillac SRX SUV', '2007 Cadillac Escalade EXT Crew Cab', '2012 Chevrolet Silverado 1500 Hybrid Crew Cab', '2012 Chevrolet Corvette Convertible', '2012 Chevrolet Corvette ZR1', '2007 Chevrolet Corvette Ron Fellows Edition Z06', '2012 Chevrolet Traverse SUV', '2012 Chevrolet Camaro Convertible', '2010 Chevrolet HHR SS', '2007 Chevrolet Impala Sedan', '2012 Chevrolet Tahoe Hybrid SUV', '2012 Chevrolet Sonic Sedan', '2007 Chevrolet Express Cargo Van', '2012 Chevrolet Avalanche Crew Cab', '2010 Chevrolet Cobalt SS', '2010 Chevrolet Malibu HybridSedan', '2009 Chevrolet TrailBlazer SS', '2012 Chevrolet Silverado 2500HD Regular Cab', '2007 Chevrolet Silverado 1500 Classic Extended Cab', '2007 Chevrolet Express Van', '2007 Chevrolet Monte CarloCoupe', '2007 Chevrolet Malibu Sedan', '2012 Chevrolet Silverado 1500 Extended Cab', '2012 Chevrolet Silverado 1500 Regular Cab', '2009 Chrysler Aspen SUV', '2010 Chrysler Sebring Convertible', '2012Chrysler Town and Country Minivan', '2010 Chrysler 300 SRT-8', '2008 Chrysler Crossfire Convertible', '2008 Chrysler PT Cruiser Convertible', '2002 Daewoo Nubira Wagon', '2012 Dodge Caliber Wagon', '2007 Dodge Caliber Wagon', '1997 Dodge Caravan Minivan', '2010 Dodge Ram Pickup 3500 Crew Cab', '2009 Dodge Ram Pickup 3500 Quad Cab', '2009 Dodge Sprinter Cargo Van', '2012 Dodge Journey SUV', '2010 Dodge Dakota Crew Cab', '2007 Dodge Dakota Club Cab', '2008 Dodge Magnum Wagon', '2011 Dodge Challenger SRT8', '2012 Dodge Durango SUV', '2007 Dodge Durango SUV', '2012 Dodge Charger Sedan', '2009 Dodge Charger SRT-8', '1998 Eagle Talon Hatchback']    
OxfordPets(19) -> 50
['abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua', 'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond']
OxfordFlowers(51) -> 100
['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia']
Caltech101(50) -> 100
['face', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter']
SUN397(199) -> 400
['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'outdoor apartment_building', 'indoor apse', 'aquarium', 'aqueduct','arch', 'archive', 'outdoor arrival_gate', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'outdoor athletic_field', 'public atrium', 'attic', 'auditorium', 'auto_factory', 'badlands', 'indoor badminton_court', 'baggage_claim', 'shop bakery', 'exterior balcony', 'interior balcony', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'outdoor basketball_court', 'bathroom', 'batters_box', 'bayou', 'indoor bazaar', 'outdoor bazaar', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'indoor bistro', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'indoor booth', 'botanical_garden', 'indoor bow_window', 'outdoor bow_window', 'bowling_alley', 'boxing_ring', 'indoor brewery', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal', 'urban canal', 'candy_store', 'canyon', 'backseat car_interior', 'frontseat car_interior', 'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral', 'indoor cavern', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'indoor chicken_coop', 'outdoor chicken_coop', 'childs_room', 'indoor church', 'outdoor church', 'classroom', 'clean_room', 'cliff', 'indoor cloister', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'outdoor control_tower', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'exterior covered_bridge', 'creek', 'crevasse', 'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists_office', 'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'home dinette', 'vehicle dinette', 'dining_car', 'dining_room', 'discotheque', 'dock', 'outdoor doorway', 'dorm_room', 'driveway', 'outdoor driving_range', 'drugstore','electrical_substation', 'door elevator', 'interior elevator', 'elevator_shaft', 'engine_room', 'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'fastfood_restaurant', 'cultivated field', 'wild field', 'fire_escape', 'fire_station', 'indoor firing_range', 'fishpond', 'indoor florist_shop', 'food_court', 'broadleaf forest', 'needleleaf forest', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'indoor garage', 'garbage_dump', 'gas_station', 'exterior gazebo', 'indoor general_store', 'outdoor general_store', 'gift_shop', 'golf_course', 'indoor greenhouse', 'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'outdoor hot_tub', 'outdoor hotel', 'hotel_room', 'house', 'outdoor hunting_lodge', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'indoor ice_skating_rink']     
UCF101(51) -> 100
['Apply_Eye_Makeup', 'Apply_Lipstick', 'Archery', 'Baby_Crawling', 'Balance_Beam', 'Band_Marching', 'Baseball_Pitch', 'Basketball', 'Basketball_Dunk', 'Bench_Press', 'Biking', 'Billiards', 'Blow_Dry_Hair', 'Blowing_Candles', 'Body_Weight_Squats', 'Bowling', 'Boxing_Punching_Bag', 'Boxing_Speed_Bag', 'Breast_Stroke', 'Brushing_Teeth', 'Clean_And_Jerk', 'Cliff_Diving', 'Cricket_Bowling', 'Cricket_Shot', 'Cutting_In_Kitchen', 'Diving', 'Drumming', 'Fencing', 'Field_Hockey_Penalty', 'Floor_Gymnastics', 'Frisbee_Catch', 'Front_Crawl', 'Golf_Swing', 'Haircut', 'Hammering', 'Hammer_Throw', 'Handstand_Pushups', 'Handstand_Walking', 'Head_Massage', 'High_Jump', 'Horse_Race', 'Horse_Riding', 'Hula_Hoop', 'Ice_Dancing', 'Javelin_Throw', 'Juggling_Balls', 'Jumping_Jack', 'Jump_Rope', 'Kayaking', 'Knitting', 'Long_Jump']
'''

VIRTUAL_CLASSNAMES = {
    "DescribableTextures": [
        'speckled', 'woven', 'knobbly', 'striated', 'pitted',
        'scaly', 'latticed', 'spotted', 'meshed', 'corrugated',
        'glossy', 'frosted', 'pebbled', 'silky', 'snaked',
        'stippling', 'tiled', 'twisted', 'braided-looped', 'perforated',
        'segmented', 'netted', 'patched', 'shimmering', 'spongy',
        'scuffed', 'splattered', 'etched', 'veined', 'puckered',
        'quilted', 'embossed', 'fuzzy', 'furry', 'stippled',
        'glistening', 'rippled', 'pattered', 'foamy', 'tasselled',
        'hollowed', 'torn', 'ribboned', 'fibred', 'shaggy',
        'wrinkled', 'fringed', 'creased', 'fractal', 'grained'
    ],
    "FGVCAircraft": [
        "707-100", "707-120", "707-700", "717-300", "720B", "727-100", "737-MAX7", "737-MAX8", "737-MAX9",
        "747-8", "747-SP", "757-100", "757-300F", "767-100", "767-F", "777F", "777-8", "777-9X", "787-8",
        "787-9", "787-10", "A220-100", "A220-300", "A300-600", "A310-300", "A320neo", "A321neo", "A330neo",
        "A330-800", "A330-900", "A350-900", "A350-1000", "AH-64 Apache", "An-24", "An-26", "An-72", "BAC 1-11",
        "BAE Jetstream 31", "Beechcraft King Air", "Beechcraft Super King Air", "Bombardier CS100",
        "Bombardier CS300", "Bombardier Dash 8", "C-17 Globemaster", "C-5 Galaxy", "Cessna 150", "Cessna 182",
        "Cessna Citation X", "Cessna Sovereign", "Challenger 300", "Challenger 600", "Cirrus SR20", "Cirrus SR22",
        "Concorde", "DHC-6 Twin Otter", "DHC-7", "DHC-8", "DC-10", "DC-8", "DC-9", "DC-3", "DC-6", "Do 228",
        "E-3 Sentry", "E170", "E175", "E190", "E195", "Embraer ERJ135", "Embraer ERJ145", "F-14 Tomcat", 
        "F-15 Eagle", "F-16 Falcon", "F/A-18 Hornet", "Fokker 100", "Fokker 50", "Fokker 70", "Fokker F27",
        "Gulfstream G200", "Gulfstream G280", "Gulfstream G450", "Gulfstream G550", "Gulfstream G650",
        "Il-76", "Il-96", "Jetstream 41", "King Air 350", "Learjet 35", "Learjet 60", "MD-11", "MD-80", 
        "MD-81", "MD-82", "MD-83", "MD-87", "MD-90", "Piaggio Avanti", "Saab 340", "Saab 2000", "Tu-154"
    ],
    "EuroSAT": [
        'Residential Area',
        'Airport Runway',
        'River or Stream',
        'Solar Farm',
        'Golf Course',
        'Bare Soil Land',
        'Railway Track',
        'Snow-Covered Area',
        'Greenhouse Facility',
        'Wind Turbine Zone'
    ],
    "StanfordCars": [
        '2013 Acura ILX Sedan', '2015 Acura NSX Coupe', '2005 Acura RSX Type-S', '2014 Acura MDX SUV', '2009 Acura RDX SUV',
        '2011 Aston Martin DB9 Coupe', '2013 Aston Martin Rapide Sedan', '2014 Aston Martin Vanquish Coupe', '2015 Aston Martin DB11 Convertible', '2016 Aston Martin Lagonda Sedan',
        '2013 Audi A7 Sportback', '2014 Audi Q5 SUV', '2016 Audi A3 Convertible', '2013 Audi S3 Sedan', '2015 Audi A6 Allroad Wagon',
        '2016 BMW 2 Series Coupe', '2013 BMW i8 Coupe', '2014 BMW 4 Series Coupe', '2015 BMW X4 SUV', '2016 BMW M2 Coupe',
        '2014 Bentley Flying Spur Sedan', '2015 Bentley Bentayga SUV', '2016 Bentley Continental GT3-R', '2015 Bentley EXP 10 Speed 6 Coupe', '2017 Bentley Azure Convertible',
        '2013 Bugatti Chiron Coupe', '2015 Bugatti Veyron Grand Sport Vitesse Convertible', '2016 Bugatti Divo Coupe', '2017 Bugatti Centodieci Coupe', '2018 Bugatti La Voiture Noire Coupe',
        '2014 Buick Encore SUV', '2015 Buick LaCrosse Sedan', '2016 Buick Cascada Convertible', '2014 Buick Lucerne Sedan', '2013 Buick Terraza Minivan',
        '2013 Cadillac ATS Coupe', '2015 Cadillac ELR Coupe', '2016 Cadillac CT6 Sedan', '2015 Cadillac XTS Sedan', '2016 Cadillac Escalade SUV',
        '2013 Chevrolet Cruze Sedan', '2014 Chevrolet SS Sedan', '2015 Chevrolet Volt Hatchback', '2016 Chevrolet Suburban SUV', '2015 Chevrolet Spark EV Hatchback',
        '2016 Chevrolet Colorado Crew Cab', '2015 Chevrolet Malibu Limited Sedan', '2014 Chevrolet Equinox SUV', '2015 Chevrolet City Express Van', '2016 Chevrolet Blazer SUV',
        '2013 Chrysler 200 Convertible', '2015 Chrysler Pacifica Minivan', '2014 Chrysler 300C Sedan', '2016 Chrysler Aspen Hybrid SUV', '2015 Chrysler Concorde Sedan',
        '2013 Daewoo Leganza Sedan', '2000 Daewoo Lanos Hatchback', '2001 Daewoo Matiz Hatchback', '2002 Daewoo Tacuma Minivan', '2003 Daewoo Rezzo Wagon',
        '2013 Dodge Avenger Sedan', '2014 Dodge Dart Sedan', '2016 Dodge Viper GTS Coupe', '2015 Dodge Grand Caravan Minivan', '2016 Dodge Nitro SUV',
        '2014 Dodge Challenger Hellcat', '2013 Dodge Neon SRT-4', '2015 Dodge Ram 1500 Rebel Crew Cab', '2016 Dodge Journey Crossroad SUV', '2016 Dodge Charger Daytona Sedan',
        '2013 Ford Focus Hatchback', '2014 Ford Fusion Energi Sedan', '2016 Ford Edge SUV', '2015 Ford Explorer Sport SUV', '2016 Ford Escape Titanium SUV',
        '2014 Ford Mustang GT Coupe', '2015 Ford F-150 Raptor Crew Cab', '2016 Ford Fiesta ST Hatchback', '2013 Ford Flex SUV', '2015 Ford C-Max Hybrid Hatchback',
        '2013 GMC Terrain SUV', '2014 GMC Acadia Denali SUV', '2016 GMC Canyon Crew Cab', '2015 GMC Yukon XL SUV', '2016 GMC Savana Passenger Van',
        '2013 Honda Fit EV Hatchback', '2014 Honda Accord Coupe', '2015 Honda Crosstour Hatchback', '2016 Honda HR-V SUV', '2016 Honda Ridgeline Pickup',
        '2014 Hyundai Elantra GT Hatchback', '2015 Hyundai Azera Sedan', '2016 Hyundai Veloster Turbo Hatchback', '2015 Hyundai Genesis Coupe', '2016 Hyundai Sonata Plug-In Hybrid',
        '2013 Infiniti G37 Convertible', '2014 Infiniti Q50 Sedan', '2016 Infiniti QX60 SUV', '2015 Infiniti QX30 Hatchback', '2016 Infiniti Q70L Sedan',
        '2013 Jaguar XF Sedan', '2015 Jaguar F-Type Coupe', '2016 Jaguar XJ Sedan', '2014 Jaguar XE Sedan', '2016 Jaguar F-Pace SUV',
        '2013 Jeep Patriot SUV', '2015 Jeep Compass SUV', '2016 Jeep Renegade Trailhawk SUV', '2014 Jeep Grand Cherokee SRT', '2016 Jeep Gladiator Pickup',
        '2013 Kia Forte Koup Coupe', '2014 Kia Optima Hybrid Sedan', '2015 Kia Soul EV Hatchback', '2016 Kia Sorento SUV', '2015 Kia Sportage SX SUV',
        '2013 Land Rover LR2 SUV', '2014 Land Rover Range Rover Evoque Coupe', '2016 Land Rover Discovery Sport SUV', '2015 Land Rover Defender 90 SUV', '2016 Land Rover Freelander 2 SUV',
        '2013 Lexus IS C Convertible', '2014 Lexus GS 350 Sedan', '2015 Lexus NX 300h SUV', '2016 Lexus RC F Coupe', '2016 Lexus RX 450h SUV',
        '2013 Lincoln MKT SUV', '2014 Lincoln MKZ Hybrid Sedan', '2015 Lincoln Navigator L SUV', '2016 Lincoln Continental Sedan', '2015 Lincoln MKC Reserve SUV',
        '2013 Maserati GranTurismo Convertible', '2014 Maserati Ghibli Sedan', '2015 Maserati Quattroporte S Q4', '2016 Maserati Levante SUV', '2015 Maserati Alfieri Coupe',
        '2013 Mazda CX-9 SUV', '2014 Mazda5 Minivan', '2015 Mazda6 i Grand Touring Sedan', '2016 Mazda CX-3 SUV', '2016 Mazda MX-5 Miata RF',
        '2013 Mercedes-Benz CLA-Class Sedan', '2014 Mercedes-Benz SLK-Class Convertible', '2015 Mercedes-Benz GLA-Class SUV', '2016 Mercedes-Benz S-Class Coupe', '2016 Mercedes-Benz GLS-Class SUV',
        '2013 MINI Roadster Convertible', '2014 MINI Clubman Wagon', '2015 MINI Paceman Hatchback', '2016 MINI Cooper S Countryman', '2015 MINI Convertible John Cooper Works',
        '2013 Mitsubishi Lancer Evolution Sedan', '2014 Mitsubishi Outlander Sport SUV', '2015 Mitsubishi Mirage Hatchback', '2016 Mitsubishi i-MiEV Hatchback', '2015 Mitsubishi Galant Sedan',
        '2013 Nissan Maxima Sedan', '2014 Nissan Juke NISMO RS Hatchback', '2015 Nissan Murano SUV', '2016 Nissan Titan XD Crew Cab', '2015 Nissan Leaf EV Hatchback',
        '2013 Pontiac G8 GXP Sedan', '2008 Pontiac Solstice Coupe', '2009 Pontiac G6 Convertible', '2006 Pontiac Torrent SUV', '2007 Pontiac Vibe Hatchback',
        '2013 Porsche Panamera GTS Sedan', '2014 Porsche Macan Turbo SUV', '2015 Porsche Cayman GT4 Coupe', '2016 Porsche 911 Targa 4S', '2016 Porsche 918 Spyder Coupe',
        '2013 Saab 9-5 Aero Sedan', '2010 Saab 9-3 SportCombi Wagon', '2009 Saab 9-7X SUV', '2008 Saab 9-2X Aero Hatchback', '2011 Saab 9-4X SUV',
        '2013 Scion FR-S Coupe', '2014 Scion xB Hatchback', '2015 Scion iQ Hatchback', '2016 Scion tC Coupe', '2015 Scion xD Hatchback',
        '2013 Subaru BRZ Coupe', '2014 Subaru XV Crosstrek Hybrid', '2015 Subaru Forester XT SUV', '2016 Subaru WRX STI Sedan', '2015 Subaru Outback 3.6R Wagon',
        '2013 Suzuki SX4 Crossover', '2010 Suzuki Kizashi Sedan', '2009 Suzuki Grand Vitara SUV', '2008 Suzuki XL-7 SUV', '2011 Suzuki Equator Crew Cab',
        '2013 Tesla Model S Sedan', '2015 Tesla Model X SUV', '2017 Tesla Model 3 Sedan', '2019 Tesla Model Y SUV', '2021 Tesla Roadster Coupe'
    ],
    "OxfordPets": [
        'siamese', 'persian', 'maine_coon', 'ragdoll', 'norwegian_forest',
        'scottish_fold', 'devon_rex', 'sphynx', 'manx', 'russian_blue',
        'munchkin', 'balinese', 'turkish_van', 'oriental_shorthair', 'cornish_rex',
        'shih_tzu', 'pomeranian', 'yorkshire_terrier', 'cavalier_king_charles', 'papillon',
        'pekingese', 'lhasa_apso', 'maltese', 'bichon_frise', 'toy_poodle',
        'miniature_schnauzer', 'whippet', 'basenji', 'akita', 'alaskan_malamute',
        'bernese_mountain_dog', 'border_collie', 'briard', 'cairn_terrier', 'dalmatian',
        'dandie_dinmont', 'doberman_pinscher', 'finnish_spitz', 'flat_coated_retriever', 'golden_retriever',
        'irish_setter', 'labrador_retriever', 'newfoundland', 'norfolk_terrier', 'old_english_sheepdog',
        'rottweiler', 'samoyed', 'tibetan_mastiff', 'weimaraner', 'wire_fox_terrier'
    ],
    "OxfordFlowers": [
        'silver lace vine', 'crimson columbine', 'desert marigold', 'mountain laurel',
        'frosted zinnia', 'wild blue flax', 'chocolate cosmos', 'blue-eyed grass',
        'sunburst coreopsis', 'crimson paintbrush', 'royal bluebell', 'black bat flower',
        'fringed bleeding heart', 'golden spider lily', 'coral honeysuckle', 'firewheel',
        'scarlet beebalm', 'prairie smoke', 'candle larkspur', 'dusky cranesbill',
        'orchid cactus', 'swamp milkweed', 'silver thistle', 'ice plant',
        'leopard lily', 'copper iris', 'crown imperial', 'trailing arbutus',
        'pale gentian', 'lavender mistflower', 'spotted joe-pye weed', 'azure monkshood',
        'snowdrop anemone', 'arctic poppy', 'meadow rue', 'sunset gaillardia',
        'crested iris', 'blue passionflower', 'twilight tulip', 'crimson catchfly',
        'fuzzy wuzzy lamb’s ear', 'red campion', 'spider orchid', 'ghost flower',
        'silver sagebrush', 'white bleeding heart', 'rocky mountain columbine', 'corydalis blue',
        'evening primrose', 'glacier lily', 'firebush', 'hummingbird mint',
        'cushion spurge', 'whirling butterflies', 'turk’s cap lily', 'jacob’s ladder',
        'purple tansy', 'prairie clover', 'cinnamon fern', 'wild lupine',
        'yellow loosestrife', 'jacaranda blossom', 'feverfew daisy', 'golden columbine',
        'salmon zinnia', 'bell heather', 'nodding onion', 'sneezeweed',
        'woolly blue curls', 'scarlet flax', 'golden ragwort', 'silver bell',
        'candy tuft', 'cherry pie flower', 'windflower', 'bog rosemary',
        'orange hawkweed', 'pink turtlehead', 'marsh marigold', 'mountain forget-me-not',
        'crown daisy', 'laceleaf', 'golden trumpet', 'rosy milkweed',
        'hairy beardtongue', 'pine barren gentian', 'baby blue eyes', 'creeping phlox',
        'golden aster', 'fairy fan flower', 'fringed gentian', 'yellow archangel',
        'starry campion', 'wild indigo', 'flame azalea', 'blue vervain',
        'red trillium', 'moss campion', 'white wood aster', 'rose campion',
        'silverleaf nightshade', 'canary creeper', 'chocolate daisy', 'wild blue phlox'
    ],
    "Caltech101": [
        'badger', 'tiger', 'tapir', 'stingray', 'rhino', 'orca', 'walrus', 'toucan', 'macaw', 'komodo_dragon',
        'banjo', 'harp', 'saxophone', 'mandolin', 'bugle', 'lyre', 'harmonica', 'tambourine', 'accordion_case', 'flute_case',
        'zeppelin', 'glider', 'biplane', 'space_shuttle', 'rocket_launcher', 'hovercraft', 'monorail', 'bullet_train', 'cable_car', 'submarine',
        'snowmobile', 'segway', 'rickshaw', 'dune_buggy', 'tricycle', 'golf_cart', 'jet_ski', 'kayak', 'scooter', 'wagon',
        'sundial', 'barometer', 'telescope', 'microscope', 'thermometer', 'compass', 'sextant', 'astrolabe', 'typewriter', 'abacus',
        'pogo_stick', 'yo_yo', 'marble', 'teddy_bear', 'toy_soldier', 'kaleidoscope', 'rubiks_cube', 'puppet', 'slinky', 'spinning_top',
        'teacup', 'wine_glass', 'goblet', 'mug', 'thermos', 'pitcher', 'chalice', 'saucer', 'espresso_machine', 'blender',
        'koala', 'jellyfish', 'starfish', 'octopus', 'narwhal', 'seahorse', 'penguin', 'meerkat', 'wombat', 'lemur',
        'lamp_post', 'mailbox', 'fire_hydrant', 'traffic_cone', 'satellite_dish', 'wind_turbine', 'vending_machine', 'arcade_machine', 'jukebox', 'pinball_machine',
        'carousel', 'roller_coaster', 'ferris_wheel', 'swing_set', 'slide', 'seesaw', 'sandbox', 'trampoline', 'kite', 'hot_air_balloon'
    ],
    "SUN397": [
        'indoor abandoned_restaurant', 'indoor abandoned_shed', 'indoor bright_restaurant',
        'indoor compact_area', 'indoor compact_closet', 'indoor compact_deck', 'indoor compact_platform',
        'indoor cozy_chamber', 'indoor cozy_kitchen', 'indoor cozy_restaurant', 'indoor elevated_area',
        'indoor elevated_platform', 'indoor elevated_zone', 'indoor hidden_arcade', 'indoor hidden_chamber',
        'indoor hidden_garden', 'indoor hidden_terrace', 'indoor historic_arcade', 'indoor historic_lounge',
        'indoor historic_office', 'indoor historic_shop', 'indoor lush_garden', 'indoor lush_terrace',
        'indoor modern_arcade', 'indoor modern_closet', 'indoor modern_gallery', 'indoor modern_kitchen',
        'indoor modern_office', 'indoor modern_platform', 'indoor modern_restaurant', 'indoor modern_shed',
        'indoor modern_shop', 'indoor modern_station', 'indoor modern_terrace', 'indoor modern_zone',
        'indoor noisy_arcade', 'indoor noisy_shop', 'indoor quiet_arcade', 'indoor quiet_gallery',
        'indoor quiet_kitchen', 'indoor quiet_lounge', 'indoor quiet_office', 'indoor quiet_restaurant',
        'indoor quiet_terrace', 'indoor spacious_area', 'indoor spacious_chamber', 'indoor spacious_closet',
        'indoor spacious_restaurant', 'indoor spacious_terrace', 'indoor sunlit_arcade', 'indoor sunlit_chamber',
        'indoor sunlit_gallery', 'indoor sunlit_garden', 'indoor sunlit_kitchen', 'indoor sunlit_lounge',
        'indoor sunlit_restaurant', 'indoor sunlit_terrace', 'indoor underground_arcade', 'indoor underground_chamber',
        'indoor underground_gallery', 'indoor underground_restaurant', 'indoor underground_shed', 'indoor underground_station',
        'mountain abandoned_shed', 'mountain bright_platform', 'mountain bright_zone', 'mountain compact_platform',
        'mountain cozy_garden', 'mountain elevated_area', 'mountain hidden_chamber', 'mountain hidden_garden',
        'mountain hidden_zone', 'mountain lush_garden', 'mountain lush_terrace', 'mountain modern_gallery',
        'mountain modern_station', 'mountain noisy_zone', 'mountain quiet_terrace', 'mountain spacious_area',
        'mountain spacious_zone', 'mountain sunlit_terrace', 'mountain underground_shed', 'mountain underground_station',
        'open bright_area', 'open bright_field', 'open bright_platform', 'open compact_area', 'open compact_garden',
        'open cozy_area', 'open cozy_garden', 'open cozy_platform', 'open hidden_garden', 'open hidden_zone',
        'open historic_terrace', 'open lush_field', 'open modern_area', 'open modern_field', 'open modern_terrace',
        'open modern_zone', 'open noisy_platform', 'open quiet_garden', 'open quiet_zone', 'open spacious_area',
        'open spacious_field', 'open spacious_garden', 'open spacious_platform', 'open sunlit_field', 'open sunlit_garden',
        'open sunlit_platform', 'open underground_platform', 'outdoor abandoned_platform', 'outdoor bright_arcade',
        'outdoor bright_chamber', 'outdoor bright_garden', 'outdoor bright_platform', 'outdoor bright_zone',
        'outdoor compact_garden', 'outdoor compact_terrace', 'outdoor cozy_arcade', 'outdoor cozy_garden',
        'outdoor cozy_platform', 'outdoor cozy_zone', 'outdoor elevated_platform', 'outdoor elevated_zone',
        'outdoor hidden_arcade', 'outdoor hidden_platform', 'outdoor historic_garden', 'outdoor historic_zone',
        'outdoor lush_garden', 'outdoor lush_zone', 'outdoor modern_arcade', 'outdoor modern_garden',
        'outdoor modern_lounge', 'outdoor modern_platform', 'outdoor modern_shop', 'outdoor modern_station',
        'outdoor modern_terrace', 'outdoor modern_zone', 'outdoor noisy_arcade', 'outdoor noisy_platform',
        'outdoor quiet_arcade', 'outdoor quiet_garden', 'outdoor quiet_zone', 'outdoor spacious_arcade',
        'outdoor spacious_garden', 'outdoor spacious_zone', 'outdoor sunlit_arcade', 'outdoor sunlit_garden',
        'outdoor sunlit_terrace', 'outdoor sunlit_zone', 'outdoor underground_platform', 'outdoor underground_zone',
        'rural abandoned_garden', 'rural abandoned_shed', 'rural bright_garden', 'rural bright_platform',
        'rural compact_shed', 'rural cozy_garden', 'rural cozy_shed', 'rural elevated_zone', 'rural hidden_garden',
        'rural historic_garden', 'rural lush_terrace', 'rural modern_garden', 'rural modern_shed', 'rural modern_station',
        'rural quiet_garden', 'rural spacious_garden', 'rural spacious_platform', 'rural sunlit_garden',
        'rural underground_shed', 'rural underground_station', 'suburban bright_garden', 'suburban compact_area',
        'suburban cozy_zone', 'suburban elevated_platform', 'suburban hidden_zone', 'suburban historic_arcade',
        'suburban historic_station', 'suburban modern_garden', 'suburban modern_lounge', 'suburban modern_zone',
        'suburban noisy_garden', 'suburban quiet_garden', 'suburban spacious_zone', 'suburban sunlit_garden',
        'suburban underground_zone', 'underground abandoned_chamber', 'underground abandoned_platform',
        'underground abandoned_station', 'underground compact_chamber', 'underground compact_zone',
        'underground cozy_chamber', 'underground cozy_platform', 'underground cozy_zone', 'underground hidden_station',
        'underground historic_chamber', 'underground historic_station', 'underground modern_chamber',
        'underground modern_station', 'underground noisy_chamber', 'underground quiet_chamber',
        'underground quiet_platform', 'underground spacious_chamber', 'underground spacious_platform',
        'underground sunlit_chamber', 'urban abandoned_arcade', 'urban abandoned_station', 'urban bright_area',
        'urban bright_station', 'urban compact_station', 'urban cozy_area', 'urban cozy_lounge', 'urban elevated_platform',
        'urban hidden_arcade', 'urban hidden_station', 'urban historic_area', 'urban historic_gallery',
        'urban lush_terrace', 'urban modern_gallery', 'urban modern_station', 'urban noisy_arcade',
        'urban quiet_station', 'urban spacious_platform', 'urban sunlit_gallery', 'urban sunlit_lounge'
    ],
    "UCF101":  [
        'Aerial_Yoga', 'Animal_Feeding', 'Arm_Wrestling', 'Axe_Throwing', 'Backflip', 'Balance_Board',
        'Ballet_Twirl', 'Barbell_Curl', 'Beach_Volleyball', 'Bench_Dips', 'Bobsled_Racing', 'Bottle_Flipping',
        'Box_Jumping', 'Bubble_Blowing', 'Candle_Making', 'Canoe_Sprint', 'Card_Shuffling', 'Cartwheel',
        'Chopping_Wood', 'Clapping', 'Climbing_Ladder', 'Coffee_Pouring', 'Cornhole_Throw', 'Cooking_Pasta',
        'Crate_Stacking', 'Crowd_Waving', 'Curling_Sport', 'Cyclocross', 'Dance_Battle', 'Dodgeball',
        'Dog_Agility', 'Drawing_Sketch', 'Dumbbell_Fly', 'Egg_Toss', 'Face_Washing', 'Fan_Waving',
        'Fire_Eating', 'Fishing_Cast', 'Flag_Waving', 'Flipping_Pancake', 'Flower_Arranging', 'Foam_Rolling',
        'Football_Catch', 'Foot_Massage', 'Glass_Blowing', 'Goggles_Adjusting', 'Golf_Putting', 'Hacky_Sack',
        'Hair_Braiding', 'Hand_Clapping', 'Handwriting', 'Hiking_Uphill', 'Hurdle_Jump', 'Ice_Skating_Spin',
        'Inline_Skating', 'Jump_High_Kick', 'Kettle_Bell_Swing', 'Kickball', 'Kite_Flying', 'Ladder_Climbing',
        'Lawn_Mowing', 'Leaf_Raking', 'Light_Saber_Duel', 'Martial_Arts_Kick', 'Massage_Back', 'Mixing_Drinks',
        'Mountain_Biking', 'Nunchuck_Spinning', 'Origami_Folding', 'Painting_Wall', 'Paper_Plane_Throw',
        'Parkour_Vault', 'Pencil_Sharpening', 'Ping_Pong_Smash', 'Pizza_Tossing', 'Plant_Watering',
        'Pole_Vault', 'Push_Cart', 'Putting_Contacts', 'Racket_Spin', 'Rock_Climbing_Indoor', 'Rollerblading',
        'Rowing_Competition', 'Rugby_Pass', 'Salsa_Dancing', 'Sand_Castle_Building', 'Scarf_Tying',
        'Scuba_Diving', 'Shaving_Beard', 'Shopping_Bag_Lift', 'Shoveling_Snow', 'Shuffle_Dance', 'Ski_Jumping',
        'Slackline_Walking', 'Snowball_Throw', 'Snowboarding', 'Soap_Bubble_Play', 'Speed_Walking', 'Spinning_Wheel',
        'Squash_Rally', 'Stretching_Legs', 'Surfing_Wave', 'Swing_Jumping', 'Sword_Fight', 'Table_Saw_Use'
    ]
    
}


IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
]

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        print(classnames)
        if cfg.TRAINER.PROMPTSRC.VIRTUAL_CLASS:
            print("Using virtual classnames")
            vrclassnames = VIRTUAL_CLASSNAMES[cfg.DATASET.NAME]
            
            # suffle with seed=0
            np.random.seed(0)
            np.random.shuffle(vrclassnames)
            
            # remove duplicates
            l = len(classnames)
            p = int(cfg.TRAINER.PROMPTSRC.VIRTUAL_CLASS_PERCENTAGE)
            v = math.ceil(l * p / 100)
            
            for name in vrclassnames:
                if len(classnames) >= l + v:
                    print(f"{l} classes + {p}% virtual classes = {len(classnames)}")
                    break
                if name not in classnames:
                    classnames.append(name)
            
            if cfg.TRAINER.PROMPTSRC.VIRTUAL_IMAGE:
                class_prompt_dict = defaultdict(list)
                device = clip_model.parameters().__next__().device
                clip_model = clip_model.cuda()
                # for name in tqdm(classnames[l:]):
                for name in tqdm(classnames):
                    name = name.replace("_", " ")
                    for template in IMAGENET_TEMPLATES:
                        text = template.format(name)
                        with torch.no_grad():
                            text_feature = clip_model.encode_text(clip.tokenize(text).cuda())
                        class_prompt_dict[name].append(text_feature)
                clip_model = clip_model.to(device)
                class_prompt_dict = {k: torch.concat(v, dim=0) for k, v in class_prompt_dict.items()}
                self.class_prompt_dict = class_prompt_dict
                
        n_cls = len(classnames)
        print(f"len={len(classnames)}, {classnames}")
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.orig_n_cls = len(classnames)
        print(f"Original number of classes: {self.orig_n_cls}")
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        # self.n_cls = self.prompt_learner.n_cls

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        if image.shape[-1] == 512: # jin added TODO
            # If the image is already encoded, we don't need to encode it again
            image_features = image
        else:
            image_features = self.image_encoder(image.type(self.dtype))  
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                if image.shape[-1] == 512: # jin added TODO
                    zero_shot_features = image
                else:
                    zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                    
                # zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
                
            return F.cross_entropy(logits, label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
            
            loss_ce = F.cross_entropy(logits[:, :self.orig_n_cls], label) # jin TODO: remove this
            # text_features = text_features[:self.orig_n_cls]
            # zero_shot_logits = zero_shot_logits[:, :self.orig_n_cls]
            # logits = logits[:, :self.orig_n_cls]
            # fixed_embeddings = fixed_embeddings[:self.orig_n_cls]
            # logits[:, :self.orig_n_cls] = logits[:, :self.orig_n_cls] * 1.5
            # zero_shot_logits = zero_shot_logits * zero_shot_logits
            # breakpoint()

            return loss_ce, text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
            # return F.cross_entropy(logits,
            #                        label), text_features, fixed_embeddings, zero_shot_features, \
            #        image_features, zero_shot_logits, logits
        else:
            return logits


@TRAINER_REGISTRY.register()
class PromptSRC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.orig_n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None
        
        if cfg.TRAINER.PROMPTSRC.VIRTUAL_IMAGE:
            from dassl.data.transforms import build_transform
            from dassl.data.data_manager import build_data_loader
            from dassl.data.datasets.base_dataset import Datum
            print("Using virtual image features")
            with torch.no_grad():
                device = clip_model.parameters().__next__().device
                clip_model = clip_model.cuda()
                img_features = [data['img'] for data in self.train_loader_x.dataset]
                img_features = torch.stack(img_features, dim=0)
                # img_features = img_features.cuda()
                # img_features = clip_model.encode_image(img_features) # to much memory
                img_features = [clip_model.encode_image(img_feat.unsqueeze(0).cuda()).cpu() for img_feat in img_features]
                img_features = torch.cat(img_features, dim=0)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                clip_model = clip_model.to(device)
                
                additional_data = []
                for i, name in enumerate(classnames):
                    # if name in self.model.prompt_learner.class_prompt_dict.keys():
                    text = self.model.prompt_learner.class_prompt_dict[name].cuda()
                    start = 0
                    interval = 100
                    projected_text = 0
                    while start < img_features.shape[0]:
                        end = start + interval
                        img = img_features[start:end].cuda()
                        weight = text @ img.T
                        projected_text += weight @ img 
                        start = end
                    projected_text = projected_text / projected_text.norm(dim=-1, keepdim=True)
                    projected_text = projected_text.cpu()
                    for j, t in enumerate(projected_text):
                        additional_data.append({'img': t, 'label': i})
                        # if j == 15:
                        #     break
                        # TODO jin: 60개 전부 다 넣는걸로 해놨는데 나중에 ablation 필요.
                        
            # extended_data_source = self.train_loader_x.dataset.data_source + additional_data
            # # extended_data_source = [[data] for data in extended_data_source]
            # self.train_loader_x = build_data_loader(
            #     cfg,
            #     sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            #     data_source=extended_data_source,
            #     batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            #     n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            #     n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            #     tfm=build_transform(cfg, is_train=True),
            #     is_train=True,
            #     dataset_wrapper=CustomDatasetWrapper
            # )
            
            self.train_loader_a = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=additional_data,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(cfg, is_train=True),
                is_train=True,
                dataset_wrapper=CustomDatasetWrapper
            )
            # class _DatasetWrapper(Dataset):
            #     def __init__(self, data_list):
            #         self.data = data_list

            #     def __len__(self):
            #         return len(self.data)

            #     def __getitem__(self, idx):
            #         return self.data[idx] 
            
            # self.train_loader_a = build_data_loader(
            #             cfg,
            #             sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            #             data_source=_DatasetWrapper(additional_data),
            #             batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            #             n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            #             n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            #             tfm=build_transform(cfg, is_train=True),
            #             is_train=True,
            #             dataset_wrapper=None, 
            #         )
            # self.train_loader_a = DataLoader(_DatasetWrapper(additional_data), batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
            # for batch in self.train_loader_a:
            #     image, label = self.parse_batch_train(batch)
            #     image = image.cuda()
            #     label = label.cuda()
                # print(image.shape, label.shape)
                # breakpoint()
                # print(image[0].shape, label[0].shape)
                # breakpoint()
        
    
    def forward_backward(self, batch, gpa=True):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            if self.cfg.TRAINER.PROMPTSRC.NO_SCL:
                L_SCL = 0
            loss = (loss_ce + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if gpa: # jin added TODO
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
                # Means one epoch is completed, perform GPA
                self.step_counter = self.step_counter + 1
                current_epoch_weight = self.gauss[self.step_counter - 2]
                current_model_weights = copy.deepcopy(model.state_dict())
                weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
                if self.previous_model_gpa is None:
                    self.previous_model_gpa = weighted_state_dict
                else:
                    self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

            if self.step_counter == self.model.total_epochs + 1:
                print("Using GPA model for final inference...")
                model.load_state_dict(self.previous_model_gpa)
                self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    def run_epoch(self):
        self.set_model_mode("train")
        
        if self.cfg.TRAINER.PROMPTSRC.VIRTUAL_IMAGE: #jin added TODO
            self.run_addtional_epoch()   
        
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()     

    def run_addtional_epoch(self):
        print("Training with virtual image features")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_a)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_a):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, False)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
            

from dassl.utils import read_image
from torch.utils.data import Dataset as TorchDataset
from dassl.data.datasets.base_dataset import Datum
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
import torchvision.transforms as T
class CustomDatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        if isinstance(item, Datum):
            output = {
                "label": item.label,
                "domain": item.domain,
                "impath": item.impath,
                "index": idx
            }

            img0 = read_image(item.impath)

            if self.transform is not None:
                if isinstance(self.transform, (list, tuple)):
                    for i, tfm in enumerate(self.transform):
                        img = self._transform_image(tfm, img0)
                        keyname = "img"
                        if (i + 1) > 1:
                            keyname += str(i + 1)
                        output[keyname] = img
                else:
                    img = self._transform_image(self.transform, img0)
                    output["img"] = img
            else:
                output["img"] = img0

            if self.return_img0:
                output["img0"] = self.to_tensor(img0)  # without any augmentation
        else:
            output = {
                "label": item["label"],
                "index": idx,
                "img": item["img"],
            }

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

