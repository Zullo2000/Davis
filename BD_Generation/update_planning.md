Change the planning_T1.md Extensibility Map (Post-v1). When we rtefer to a specific model follow the paper and its github code.

This is what we will do after v1 (a-la-MDLM), but only after having a good evaluation pipeline:

- remasking with Max-Capped Schedule (ReMDM): trying eta = [0.1, 0.2, 0.3, 0.4, 0.5]. No need to re-train the model.
- remasking with Confidence-Based Schedule (ReMDM): with ‘confidence score’ for unmasked tokens. No need to re-train the model.
- learning also the forward process (MELD) to avoid state-clashing in edges
- doing constrained generation using guidance. example of constrains we care about: BD_Generation\constrains_examples.md
