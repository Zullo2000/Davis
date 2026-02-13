# Davis: Discrete Diffusion for Residential Floorplan Generation

Davis is a research project focused on generating residential floorplan layouts using **discrete diffusion models** applied to graph structures. The codebase and its documentation are addressed to a smart use of Claude Code and Codex.

## Overview

The goal of this project is to automatically generate feasible and diverse residential floorplans (bubble diagrams) that respect architectural constraints and user preferences. Rather than traditional autoregressive approaches, we leverage the power of discrete diffusion models to enable:

- **Parallel generation**: All rooms can be generated simultaneously rather than sequentially
- **Order-agnostic modeling**: Graphs naturally represent unordered data (rooms and their relationships)
- **Controllable generation**: Diffusion models allow for guided sampling and constraint satisfaction
- **Geometric validity**: Ensuring spatial relationships are architecturally sound

## Technical Approach

### Graph Representation

We represent floorplans as **bubble diagrams** (layout graphs):
- **Nodes**: Rooms with attributes (type, size, location)
- **Edges**: Spatial relationships between rooms (adjacency, inside/surrounding)

This graph-based representation captures the topological structure of a floorplan before committing to precise geometric coordinates.

### Discrete Diffusion Models

Unlike continuous diffusion (e.g., DDPM for images), we use **discrete categorical diffusion** suitable for graph generation:

1. **Forward process**: Gradually corrupts the clean graph by randomly masking or replacing node/edge categories
2. **Reverse process**: A neural network learns to denoise the corrupted graph, iteratively refining it back to a valid floorplan

This approach is inspired by models like:
- **D3PM** (Discrete Denoising Diffusion Probabilistic Models)
- **MDLM** (Masked Diffusion Language Models)
- **GraphARM** (Autoregressive Diffusion Models for graphs)

### Why Discrete Diffusion?

**Advantages over autoregressive models**:
- No error propagation across sequential generation steps
- Can perform validity checks at each denoising step
- Naturally handles variable-length graphs (different numbers of rooms)
- Supports parallel decoding

**Advantages over GANs**:
- More stable training
- Better diversity in generated samples
- Easier to incorporate hard constraints

## Dataset

The project uses the **RPLAN dataset**, a large-scale collection of ~86,000 residential floorplans with:
- Room bounding boxes and types (bedroom, kitchen, bathroom, etc.)
- Adjacency relationships between rooms
- Building boundary polygons

## Project Structure

```
Davis/
├── BD_Generation/          # Main implementation (Bubble Diagram generation)
├── DiDAPS_COPY/           # Reference implementations (noise schedules, adaLN)
├── Papers/                # Literature review on discrete diffusion
├── Data/                  # RPLAN dataset exploration
└── README.md
```

## Implementation Status

This project is currently in the implementation phase. See `BD_Generation/` for the main codebase and `planning_T1.md` for the detailed technical specification.

## Key References

- **Graph2Plan**: Learning Floorplan Generation from Layout Graphs
- **HouseDiffusion**: Vector Floorplan Generation via Diffusion with Discrete and Continuous Denoising
- **GSDiff**: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation
- **GraphARM**: Autoregressive Diffusion Models for graph generation
- **D3PM**: Structured Denoising Diffusion Models in Discrete State-Spaces

## Research Goals

Our research aims to:
1. Generate architecturally valid bubble diagrams that satisfy spatial constraints
2. Support controllable generation (user-specified room counts, adjacencies, programs)
3. Ensure diversity in generated layouts under the same constraints
4. Enable iterative refinement and local edits to existing designs

---

*This project explores the intersection of generative AI, architectural design automation, and discrete probabilistic modeling.*
