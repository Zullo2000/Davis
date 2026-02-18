POSSIBLE CONSTRAINS:

Required program:
Every plan must contain exactly one living room and exactly one kitchen.

Bedroom count:
The plan must contain between 1 and 4 bedrooms.

Bathroom provision:
The plan must contain at least 1 bathroom.
If there are 3 or 4 bedrooms, the plan must contain at least 2 bathrooms.

Connectivity:
All interior rooms must belong to a single connected component in the room adjacency graph (no isolated rooms).

Kitchen adjacency:
The kitchen must be adjacent to the living room.

Bedroom access:
No bedroom may be accessible only through another bedroom
(i.e., a bedroom should have a non-bedroom neighbor on at least one path to the living room).

Bathroom–kitchen separation:
No bathroom may share a wall with the kitchen.

Exterior exposure for habitable rooms:
The living room must touch the exterior boundary.
At least 70% of bedrooms must touch the exterior boundary.

Kitchen ventilation proxy:
The kitchen must touch the exterior boundary or be adjacent to a balcony.

Living room minimum size:
The living room must be at least 18% of the total interior area.

Bedroom minimum size:
Each bedroom must be at least 8% of the total interior area.

Kitchen size band:
The kitchen must be between 6% and 14% of the total interior area.

Bathroom size band:
Each bathroom must be between 2.5% and 8% of the total interior area.

Balcony rules:
If a balcony exists, it must touch the exterior boundary.
The total balcony area must be at most 15% of the total interior area.

Aspect ratio sanity:
No room may have a bounding-box aspect ratio greater than 4:1.