"""
Comprehensive database of Indian cities with coordinates.

Each city has: name, state, latitude, longitude.
Bounding boxes are generated dynamically based on city size tier.
"""

from typing import Optional

# (name, state, lat, lon, tier)
# tier: 1 = mega city (large bbox), 2 = metro, 3 = smaller city
INDIAN_CITIES: list[dict] = [
    # Andhra Pradesh
    {"name": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "tier": 2},
    {"name": "Vijayawada", "state": "Andhra Pradesh", "lat": 16.5062, "lon": 80.6480, "tier": 3},
    {"name": "Guntur", "state": "Andhra Pradesh", "lat": 16.3067, "lon": 80.4365, "tier": 3},
    {"name": "Nellore", "state": "Andhra Pradesh", "lat": 14.4426, "lon": 79.9865, "tier": 3},
    {"name": "Kurnool", "state": "Andhra Pradesh", "lat": 15.8281, "lon": 78.0373, "tier": 3},
    {"name": "Rajahmundry", "state": "Andhra Pradesh", "lat": 17.0005, "lon": 81.8040, "tier": 3},
    {"name": "Tirupati", "state": "Andhra Pradesh", "lat": 13.6288, "lon": 79.4192, "tier": 3},
    {"name": "Kakinada", "state": "Andhra Pradesh", "lat": 16.9891, "lon": 82.2475, "tier": 3},
    {"name": "Anantapur", "state": "Andhra Pradesh", "lat": 14.6819, "lon": 77.6006, "tier": 3},
    {"name": "Amaravati", "state": "Andhra Pradesh", "lat": 16.5131, "lon": 80.5150, "tier": 3},
    # Arunachal Pradesh
    {"name": "Itanagar", "state": "Arunachal Pradesh", "lat": 27.0844, "lon": 93.6053, "tier": 3},
    # Assam
    {"name": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "tier": 2},
    {"name": "Silchar", "state": "Assam", "lat": 24.8333, "lon": 92.7789, "tier": 3},
    {"name": "Dibrugarh", "state": "Assam", "lat": 27.4728, "lon": 94.9120, "tier": 3},
    {"name": "Jorhat", "state": "Assam", "lat": 26.7509, "lon": 94.2037, "tier": 3},
    {"name": "Nagaon", "state": "Assam", "lat": 26.3464, "lon": 92.6840, "tier": 3},
    {"name": "Tinsukia", "state": "Assam", "lat": 27.4922, "lon": 95.3547, "tier": 3},
    # Bihar
    {"name": "Patna", "state": "Bihar", "lat": 25.6093, "lon": 85.1376, "tier": 2},
    {"name": "Gaya", "state": "Bihar", "lat": 24.7955, "lon": 84.9994, "tier": 3},
    {"name": "Bhagalpur", "state": "Bihar", "lat": 25.2425, "lon": 86.9842, "tier": 3},
    {"name": "Muzaffarpur", "state": "Bihar", "lat": 26.1209, "lon": 85.3647, "tier": 3},
    {"name": "Darbhanga", "state": "Bihar", "lat": 26.1542, "lon": 85.8918, "tier": 3},
    {"name": "Purnia", "state": "Bihar", "lat": 25.7771, "lon": 87.4753, "tier": 3},
    # Chandigarh
    {"name": "Chandigarh", "state": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "tier": 2},
    # Chhattisgarh
    {"name": "Raipur", "state": "Chhattisgarh", "lat": 21.2514, "lon": 81.6296, "tier": 2},
    {"name": "Bhilai", "state": "Chhattisgarh", "lat": 21.2167, "lon": 81.4333, "tier": 3},
    {"name": "Bilaspur", "state": "Chhattisgarh", "lat": 22.0797, "lon": 82.1409, "tier": 3},
    {"name": "Korba", "state": "Chhattisgarh", "lat": 22.3595, "lon": 82.7501, "tier": 3},
    # Delhi
    {"name": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "tier": 1},
    {"name": "New Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "tier": 1},
    # Goa
    {"name": "Goa", "state": "Goa", "lat": 15.4909, "lon": 73.8278, "tier": 3},
    {"name": "Panaji", "state": "Goa", "lat": 15.4989, "lon": 73.8278, "tier": 3},
    {"name": "Margao", "state": "Goa", "lat": 15.2832, "lon": 73.9862, "tier": 3},
    {"name": "Vasco da Gama", "state": "Goa", "lat": 15.3982, "lon": 73.8113, "tier": 3},
    # Gujarat
    {"name": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "tier": 1},
    {"name": "Surat", "state": "Gujarat", "lat": 21.1702, "lon": 72.8311, "tier": 2},
    {"name": "Vadodara", "state": "Gujarat", "lat": 22.3072, "lon": 73.1812, "tier": 2},
    {"name": "Rajkot", "state": "Gujarat", "lat": 22.3039, "lon": 70.8022, "tier": 2},
    {"name": "Bhavnagar", "state": "Gujarat", "lat": 21.7645, "lon": 72.1519, "tier": 3},
    {"name": "Jamnagar", "state": "Gujarat", "lat": 22.4707, "lon": 70.0577, "tier": 3},
    {"name": "Junagadh", "state": "Gujarat", "lat": 21.5222, "lon": 70.4579, "tier": 3},
    {"name": "Gandhinagar", "state": "Gujarat", "lat": 23.2156, "lon": 72.6369, "tier": 3},
    {"name": "Anand", "state": "Gujarat", "lat": 22.5645, "lon": 72.9289, "tier": 3},
    {"name": "Morbi", "state": "Gujarat", "lat": 22.8120, "lon": 70.8376, "tier": 3},
    # Haryana
    {"name": "Faridabad", "state": "Haryana", "lat": 28.4089, "lon": 77.3178, "tier": 2},
    {"name": "Gurugram", "state": "Haryana", "lat": 28.4595, "lon": 77.0266, "tier": 2},
    {"name": "Panipat", "state": "Haryana", "lat": 29.3909, "lon": 76.9635, "tier": 3},
    {"name": "Ambala", "state": "Haryana", "lat": 30.3782, "lon": 76.7767, "tier": 3},
    {"name": "Yamunanagar", "state": "Haryana", "lat": 30.1290, "lon": 77.2674, "tier": 3},
    {"name": "Rohtak", "state": "Haryana", "lat": 28.8955, "lon": 76.6066, "tier": 3},
    {"name": "Hisar", "state": "Haryana", "lat": 29.1492, "lon": 75.7217, "tier": 3},
    {"name": "Karnal", "state": "Haryana", "lat": 29.6857, "lon": 76.9905, "tier": 3},
    {"name": "Sonipat", "state": "Haryana", "lat": 28.9845, "lon": 77.0151, "tier": 3},
    # Himachal Pradesh
    {"name": "Shimla", "state": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734, "tier": 3},
    {"name": "Dharamshala", "state": "Himachal Pradesh", "lat": 32.2190, "lon": 76.3234, "tier": 3},
    {"name": "Manali", "state": "Himachal Pradesh", "lat": 32.2396, "lon": 77.1887, "tier": 3},
    # Jharkhand
    {"name": "Ranchi", "state": "Jharkhand", "lat": 23.3441, "lon": 85.3096, "tier": 2},
    {"name": "Jamshedpur", "state": "Jharkhand", "lat": 22.8046, "lon": 86.2029, "tier": 2},
    {"name": "Dhanbad", "state": "Jharkhand", "lat": 23.7957, "lon": 86.4304, "tier": 2},
    {"name": "Bokaro", "state": "Jharkhand", "lat": 23.6693, "lon": 86.1511, "tier": 3},
    {"name": "Hazaribagh", "state": "Jharkhand", "lat": 23.9966, "lon": 85.3637, "tier": 3},
    # Karnataka
    {"name": "Bangalore", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "tier": 1},
    {"name": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "tier": 1},
    {"name": "Mysore", "state": "Karnataka", "lat": 12.2958, "lon": 76.6394, "tier": 2},
    {"name": "Mysuru", "state": "Karnataka", "lat": 12.2958, "lon": 76.6394, "tier": 2},
    {"name": "Hubli", "state": "Karnataka", "lat": 15.3647, "lon": 75.1240, "tier": 3},
    {"name": "Mangalore", "state": "Karnataka", "lat": 12.9141, "lon": 74.8560, "tier": 3},
    {"name": "Belgaum", "state": "Karnataka", "lat": 15.8497, "lon": 74.4977, "tier": 3},
    {"name": "Gulbarga", "state": "Karnataka", "lat": 17.3297, "lon": 76.8343, "tier": 3},
    {"name": "Davangere", "state": "Karnataka", "lat": 14.4644, "lon": 75.9218, "tier": 3},
    {"name": "Bellary", "state": "Karnataka", "lat": 15.1394, "lon": 76.9214, "tier": 3},
    {"name": "Shimoga", "state": "Karnataka", "lat": 13.9299, "lon": 75.5681, "tier": 3},
    # Kerala
    {"name": "Thiruvananthapuram", "state": "Kerala", "lat": 8.5241, "lon": 76.9366, "tier": 2},
    {"name": "Kochi", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "tier": 2},
    {"name": "Kozhikode", "state": "Kerala", "lat": 11.2588, "lon": 75.7804, "tier": 2},
    {"name": "Thrissur", "state": "Kerala", "lat": 10.5276, "lon": 76.2144, "tier": 3},
    {"name": "Kollam", "state": "Kerala", "lat": 8.8932, "lon": 76.6141, "tier": 3},
    {"name": "Alappuzha", "state": "Kerala", "lat": 9.4981, "lon": 76.3388, "tier": 3},
    {"name": "Kannur", "state": "Kerala", "lat": 11.8745, "lon": 75.3704, "tier": 3},
    {"name": "Palakkad", "state": "Kerala", "lat": 10.7867, "lon": 76.6548, "tier": 3},
    # Madhya Pradesh
    {"name": "Bhopal", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "tier": 2},
    {"name": "Indore", "state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577, "tier": 2},
    {"name": "Jabalpur", "state": "Madhya Pradesh", "lat": 23.1815, "lon": 79.9864, "tier": 2},
    {"name": "Gwalior", "state": "Madhya Pradesh", "lat": 26.2183, "lon": 78.1828, "tier": 2},
    {"name": "Ujjain", "state": "Madhya Pradesh", "lat": 23.1793, "lon": 75.7849, "tier": 3},
    {"name": "Sagar", "state": "Madhya Pradesh", "lat": 23.8388, "lon": 78.7378, "tier": 3},
    {"name": "Dewas", "state": "Madhya Pradesh", "lat": 22.9623, "lon": 76.0508, "tier": 3},
    {"name": "Satna", "state": "Madhya Pradesh", "lat": 24.5854, "lon": 80.8322, "tier": 3},
    {"name": "Ratlam", "state": "Madhya Pradesh", "lat": 23.3315, "lon": 75.0367, "tier": 3},
    # Maharashtra
    {"name": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "tier": 1},
    {"name": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "tier": 1},
    {"name": "Nagpur", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "tier": 2},
    {"name": "Thane", "state": "Maharashtra", "lat": 19.2183, "lon": 72.9781, "tier": 2},
    {"name": "Nashik", "state": "Maharashtra", "lat": 20.0063, "lon": 73.7900, "tier": 2},
    {"name": "Aurangabad", "state": "Maharashtra", "lat": 19.8762, "lon": 75.3433, "tier": 2},
    {"name": "Solapur", "state": "Maharashtra", "lat": 17.6599, "lon": 75.9064, "tier": 3},
    {"name": "Kolhapur", "state": "Maharashtra", "lat": 16.7050, "lon": 74.2433, "tier": 3},
    {"name": "Navi Mumbai", "state": "Maharashtra", "lat": 19.0330, "lon": 73.0297, "tier": 2},
    {"name": "Amravati", "state": "Maharashtra", "lat": 20.9320, "lon": 77.7523, "tier": 3},
    {"name": "Sangli", "state": "Maharashtra", "lat": 16.8524, "lon": 74.5815, "tier": 3},
    {"name": "Akola", "state": "Maharashtra", "lat": 20.7002, "lon": 77.0082, "tier": 3},
    {"name": "Latur", "state": "Maharashtra", "lat": 18.4088, "lon": 76.5604, "tier": 3},
    {"name": "Chandrapur", "state": "Maharashtra", "lat": 19.9615, "lon": 79.2961, "tier": 3},
    # Manipur
    {"name": "Imphal", "state": "Manipur", "lat": 24.8170, "lon": 93.9368, "tier": 3},
    # Meghalaya
    {"name": "Shillong", "state": "Meghalaya", "lat": 25.5788, "lon": 91.8933, "tier": 3},
    # Mizoram
    {"name": "Aizawl", "state": "Mizoram", "lat": 23.7271, "lon": 92.7176, "tier": 3},
    # Nagaland
    {"name": "Kohima", "state": "Nagaland", "lat": 25.6751, "lon": 94.1086, "tier": 3},
    {"name": "Dimapur", "state": "Nagaland", "lat": 25.9042, "lon": 93.7266, "tier": 3},
    # Odisha
    {"name": "Bhubaneswar", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "tier": 2},
    {"name": "Cuttack", "state": "Odisha", "lat": 20.4625, "lon": 85.8830, "tier": 3},
    {"name": "Rourkela", "state": "Odisha", "lat": 22.2604, "lon": 84.8536, "tier": 3},
    {"name": "Berhampur", "state": "Odisha", "lat": 19.3150, "lon": 84.7941, "tier": 3},
    {"name": "Sambalpur", "state": "Odisha", "lat": 21.4669, "lon": 83.9812, "tier": 3},
    # Punjab
    {"name": "Ludhiana", "state": "Punjab", "lat": 30.9010, "lon": 75.8573, "tier": 2},
    {"name": "Amritsar", "state": "Punjab", "lat": 31.6340, "lon": 74.8723, "tier": 2},
    {"name": "Jalandhar", "state": "Punjab", "lat": 31.3260, "lon": 75.5762, "tier": 2},
    {"name": "Patiala", "state": "Punjab", "lat": 30.3398, "lon": 76.3869, "tier": 3},
    {"name": "Bathinda", "state": "Punjab", "lat": 30.2110, "lon": 74.9455, "tier": 3},
    {"name": "Mohali", "state": "Punjab", "lat": 30.7046, "lon": 76.7179, "tier": 3},
    # Rajasthan
    {"name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "tier": 1},
    {"name": "Jodhpur", "state": "Rajasthan", "lat": 26.2389, "lon": 73.0243, "tier": 2},
    {"name": "Udaipur", "state": "Rajasthan", "lat": 24.5854, "lon": 73.7125, "tier": 2},
    {"name": "Kota", "state": "Rajasthan", "lat": 25.2138, "lon": 75.8648, "tier": 2},
    {"name": "Bikaner", "state": "Rajasthan", "lat": 28.0229, "lon": 73.3119, "tier": 3},
    {"name": "Ajmer", "state": "Rajasthan", "lat": 26.4499, "lon": 74.6399, "tier": 3},
    {"name": "Bhilwara", "state": "Rajasthan", "lat": 25.3407, "lon": 74.6313, "tier": 3},
    {"name": "Alwar", "state": "Rajasthan", "lat": 27.5530, "lon": 76.6346, "tier": 3},
    {"name": "Sikar", "state": "Rajasthan", "lat": 27.6094, "lon": 75.1399, "tier": 3},
    # Sikkim
    {"name": "Gangtok", "state": "Sikkim", "lat": 27.3389, "lon": 88.6065, "tier": 3},
    # Tamil Nadu
    {"name": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "tier": 1},
    {"name": "Coimbatore", "state": "Tamil Nadu", "lat": 11.0168, "lon": 76.9558, "tier": 2},
    {"name": "Madurai", "state": "Tamil Nadu", "lat": 9.9252, "lon": 78.1198, "tier": 2},
    {"name": "Tiruchirappalli", "state": "Tamil Nadu", "lat": 10.8261, "lon": 78.6829, "tier": 2},
    {"name": "Salem", "state": "Tamil Nadu", "lat": 11.6643, "lon": 78.1460, "tier": 2},
    {"name": "Tirunelveli", "state": "Tamil Nadu", "lat": 8.7139, "lon": 77.7567, "tier": 3},
    {"name": "Erode", "state": "Tamil Nadu", "lat": 11.3410, "lon": 77.7172, "tier": 3},
    {"name": "Vellore", "state": "Tamil Nadu", "lat": 12.9165, "lon": 79.1325, "tier": 3},
    {"name": "Thoothukudi", "state": "Tamil Nadu", "lat": 8.7642, "lon": 78.1348, "tier": 3},
    {"name": "Thanjavur", "state": "Tamil Nadu", "lat": 10.7870, "lon": 79.1378, "tier": 3},
    {"name": "Dindigul", "state": "Tamil Nadu", "lat": 10.3624, "lon": 77.9695, "tier": 3},
    {"name": "Tiruppur", "state": "Tamil Nadu", "lat": 11.1085, "lon": 77.3411, "tier": 3},
    {"name": "Nagercoil", "state": "Tamil Nadu", "lat": 8.1833, "lon": 77.4119, "tier": 3},
    {"name": "Cuddalore", "state": "Tamil Nadu", "lat": 11.7480, "lon": 79.7714, "tier": 3},
    {"name": "Hosur", "state": "Tamil Nadu", "lat": 12.7409, "lon": 77.8253, "tier": 3},
    # Telangana
    {"name": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "tier": 1},
    {"name": "Warangal", "state": "Telangana", "lat": 17.9784, "lon": 79.5941, "tier": 3},
    {"name": "Nizamabad", "state": "Telangana", "lat": 18.6725, "lon": 78.0941, "tier": 3},
    {"name": "Karimnagar", "state": "Telangana", "lat": 18.4386, "lon": 79.1288, "tier": 3},
    {"name": "Khammam", "state": "Telangana", "lat": 17.2473, "lon": 80.1514, "tier": 3},
    # Tripura
    {"name": "Agartala", "state": "Tripura", "lat": 23.8315, "lon": 91.2868, "tier": 3},
    # Uttar Pradesh
    {"name": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "tier": 1},
    {"name": "Kanpur", "state": "Uttar Pradesh", "lat": 26.4499, "lon": 80.3319, "tier": 1},
    {"name": "Agra", "state": "Uttar Pradesh", "lat": 27.1767, "lon": 78.0081, "tier": 2},
    {"name": "Varanasi", "state": "Uttar Pradesh", "lat": 25.3176, "lon": 82.9739, "tier": 2},
    {"name": "Prayagraj", "state": "Uttar Pradesh", "lat": 25.4358, "lon": 81.8463, "tier": 2},
    {"name": "Allahabad", "state": "Uttar Pradesh", "lat": 25.4358, "lon": 81.8463, "tier": 2},
    {"name": "Meerut", "state": "Uttar Pradesh", "lat": 28.9845, "lon": 77.7064, "tier": 2},
    {"name": "Noida", "state": "Uttar Pradesh", "lat": 28.5355, "lon": 77.3910, "tier": 2},
    {"name": "Ghaziabad", "state": "Uttar Pradesh", "lat": 28.6692, "lon": 77.4538, "tier": 2},
    {"name": "Bareilly", "state": "Uttar Pradesh", "lat": 28.3670, "lon": 79.4304, "tier": 3},
    {"name": "Aligarh", "state": "Uttar Pradesh", "lat": 27.8974, "lon": 78.0880, "tier": 3},
    {"name": "Moradabad", "state": "Uttar Pradesh", "lat": 28.8386, "lon": 78.7733, "tier": 3},
    {"name": "Gorakhpur", "state": "Uttar Pradesh", "lat": 26.7606, "lon": 83.3732, "tier": 3},
    {"name": "Saharanpur", "state": "Uttar Pradesh", "lat": 29.9680, "lon": 77.5510, "tier": 3},
    {"name": "Jhansi", "state": "Uttar Pradesh", "lat": 25.4484, "lon": 78.5685, "tier": 3},
    {"name": "Mathura", "state": "Uttar Pradesh", "lat": 27.4924, "lon": 77.6737, "tier": 3},
    {"name": "Firozabad", "state": "Uttar Pradesh", "lat": 27.1591, "lon": 78.3957, "tier": 3},
    {"name": "Ayodhya", "state": "Uttar Pradesh", "lat": 26.7922, "lon": 82.1998, "tier": 3},
    {"name": "Greater Noida", "state": "Uttar Pradesh", "lat": 28.4744, "lon": 77.5040, "tier": 2},
    # Uttarakhand
    {"name": "Dehradun", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322, "tier": 2},
    {"name": "Haridwar", "state": "Uttarakhand", "lat": 29.9457, "lon": 78.1642, "tier": 3},
    {"name": "Rishikesh", "state": "Uttarakhand", "lat": 30.0869, "lon": 78.2676, "tier": 3},
    {"name": "Haldwani", "state": "Uttarakhand", "lat": 29.2183, "lon": 79.5130, "tier": 3},
    {"name": "Roorkee", "state": "Uttarakhand", "lat": 29.8543, "lon": 77.8880, "tier": 3},
    # West Bengal
    {"name": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "tier": 1},
    {"name": "Howrah", "state": "West Bengal", "lat": 22.5958, "lon": 88.2636, "tier": 2},
    {"name": "Durgapur", "state": "West Bengal", "lat": 23.5204, "lon": 87.3119, "tier": 3},
    {"name": "Asansol", "state": "West Bengal", "lat": 23.6739, "lon": 86.9524, "tier": 3},
    {"name": "Siliguri", "state": "West Bengal", "lat": 26.7271, "lon": 88.3953, "tier": 3},
    {"name": "Bardhaman", "state": "West Bengal", "lat": 23.2324, "lon": 87.8615, "tier": 3},
    {"name": "Kharagpur", "state": "West Bengal", "lat": 22.3460, "lon": 87.2320, "tier": 3},
    # Jammu & Kashmir
    {"name": "Srinagar", "state": "Jammu & Kashmir", "lat": 34.0837, "lon": 74.7973, "tier": 2},
    {"name": "Jammu", "state": "Jammu & Kashmir", "lat": 32.7266, "lon": 74.8570, "tier": 2},
    # Ladakh
    {"name": "Leh", "state": "Ladakh", "lat": 34.1526, "lon": 77.5771, "tier": 3},
    # Puducherry
    {"name": "Puducherry", "state": "Puducherry", "lat": 11.9416, "lon": 79.8083, "tier": 3},
    {"name": "Pondicherry", "state": "Puducherry", "lat": 11.9416, "lon": 79.8083, "tier": 3},
]

# Bbox radius in degrees based on city tier
_TIER_RADIUS = {
    1: 0.25,   # ~28 km — mega cities
    2: 0.15,   # ~17 km — metros
    3: 0.10,   # ~11 km — smaller cities
}


def generate_bbox(lat: float, lon: float, tier: int = 3) -> tuple[float, float, float, float]:
    """Generate a bounding box (min_lat, min_lon, max_lat, max_lon) from center + tier."""
    r = _TIER_RADIUS.get(tier, 0.10)
    return (lat - r, lon - r, lat + r, lon + r)


def search_cities(query: str, limit: int = 15) -> list[dict]:
    """
    Search Indian cities by name prefix (case-insensitive).

    Returns list of matching cities with name, state, lat, lon, bbox.
    """
    q = query.strip().lower()
    if not q:
        return []

    # Exact prefix matches first, then substring matches
    prefix_matches = []
    substring_matches = []

    seen = set()  # de-duplicate by (lat, lon) rounded
    for city in INDIAN_CITIES:
        key = (round(city["lat"], 2), round(city["lon"], 2))
        name_lower = city["name"].lower()

        if key in seen:
            continue

        if name_lower.startswith(q):
            prefix_matches.append(city)
            seen.add(key)
        elif q in name_lower or q in city["state"].lower():
            substring_matches.append(city)
            seen.add(key)

    results = (prefix_matches + substring_matches)[:limit]

    return [
        {
            "name": c["name"],
            "state": c["state"],
            "lat": c["lat"],
            "lon": c["lon"],
            "bbox": generate_bbox(c["lat"], c["lon"], c["tier"]),
        }
        for c in results
    ]


def get_city_bbox(city_name: str) -> Optional[tuple[float, float, float, float]]:
    """
    Look up a city by exact name (case-insensitive) and return its bbox.
    Returns None if not found.
    """
    q = city_name.strip().lower()
    for city in INDIAN_CITIES:
        if city["name"].lower() == q:
            return generate_bbox(city["lat"], city["lon"], city["tier"])
    return None


def get_city_center(city_name: str) -> Optional[tuple[float, float]]:
    """Look up city center coordinates. Returns (lat, lon) or None."""
    q = city_name.strip().lower()
    for city in INDIAN_CITIES:
        if city["name"].lower() == q:
            return (city["lat"], city["lon"])
    return None
