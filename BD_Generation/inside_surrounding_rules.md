# Forbidden Inside/Surrounding Connections

**Metric:** `inside_validity = 1 - (samples with at least one forbidden pair) / total_samples`

Convention: edge `(i, j, 4)` = "i inside j"; edge `(i, j, 5)` = "i surrounds j". Each forbidden "A inside B" equally forbids "B surrounding A". Wall-in (12) is excluded from all constraints.

## Forbidden Pairs (A inside B)

| A idx | A name | B idx | B name |
|------:|--------|------:|--------|
| 0 | LivingRoom | 1 | MasterRoom |
| 0 | LivingRoom | 2 | Kitchen |
| 0 | LivingRoom | 3 | Bathroom |
| 0 | LivingRoom | 4 | DiningRoom |
| 0 | LivingRoom | 5 | ChildRoom |
| 0 | LivingRoom | 6 | StudyRoom |
| 0 | LivingRoom | 7 | SecondRoom |
| 0 | LivingRoom | 8 | GuestRoom |
| 0 | LivingRoom | 9 | Balcony |
| 0 | LivingRoom | 10 | Entrance |
| 0 | LivingRoom | 11 | Storage |
| 1 | MasterRoom | 0 | LivingRoom |
| 1 | MasterRoom | 2 | Kitchen |
| 1 | MasterRoom | 3 | Bathroom |
| 1 | MasterRoom | 4 | DiningRoom |
| 1 | MasterRoom | 5 | ChildRoom |
| 1 | MasterRoom | 6 | StudyRoom |
| 1 | MasterRoom | 7 | SecondRoom |
| 1 | MasterRoom | 8 | GuestRoom |
| 1 | MasterRoom | 9 | Balcony |
| 1 | MasterRoom | 10 | Entrance |
| 1 | MasterRoom | 11 | Storage |
| 2 | Kitchen | 1 | MasterRoom |
| 2 | Kitchen | 3 | Bathroom |
| 2 | Kitchen | 5 | ChildRoom |
| 2 | Kitchen | 6 | StudyRoom |
| 2 | Kitchen | 7 | SecondRoom |
| 2 | Kitchen | 8 | GuestRoom |
| 2 | Kitchen | 9 | Balcony |
| 2 | Kitchen | 10 | Entrance |
| 2 | Kitchen | 11 | Storage |
| 3 | Bathroom | 10 | Entrance |
| 4 | DiningRoom | 1 | MasterRoom |
| 4 | DiningRoom | 3 | Bathroom |
| 4 | DiningRoom | 5 | ChildRoom |
| 4 | DiningRoom | 8 | GuestRoom |
| 4 | DiningRoom | 9 | Balcony |
| 4 | DiningRoom | 10 | Entrance |
| 4 | DiningRoom | 11 | Storage |
| 5 | ChildRoom | 2 | Kitchen |
| 5 | ChildRoom | 3 | Bathroom |
| 5 | ChildRoom | 4 | DiningRoom |
| 5 | ChildRoom | 9 | Balcony |
| 5 | ChildRoom | 10 | Entrance |
| 5 | ChildRoom | 11 | Storage |
| 6 | StudyRoom | 3 | Bathroom |
| 6 | StudyRoom | 9 | Balcony |
| 6 | StudyRoom | 10 | Entrance |
| 7 | SecondRoom | 3 | Bathroom |
| 7 | SecondRoom | 9 | Balcony |
| 7 | SecondRoom | 10 | Entrance |
| 8 | GuestRoom | 3 | Bathroom |
| 8 | GuestRoom | 9 | Balcony |
| 8 | GuestRoom | 10 | Entrance |
| 9 | Balcony | 2 | Kitchen |
| 9 | Balcony | 3 | Bathroom |
| 9 | Balcony | 6 | StudyRoom |
| 9 | Balcony | 10 | Entrance |
| 9 | Balcony | 11 | Storage |
| 10 | Entrance | 1 | MasterRoom |
| 10 | Entrance | 2 | Kitchen |
| 10 | Entrance | 3 | Bathroom |
| 10 | Entrance | 4 | DiningRoom |
| 10 | Entrance | 5 | ChildRoom |
| 10 | Entrance | 6 | StudyRoom |
| 10 | Entrance | 7 | SecondRoom |
| 10 | Entrance | 8 | GuestRoom |
| 10 | Entrance | 9 | Balcony |
| 10 | Entrance | 11 | Storage |
| 11 | Storage | 9 | Balcony |
