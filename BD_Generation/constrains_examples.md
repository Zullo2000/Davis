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