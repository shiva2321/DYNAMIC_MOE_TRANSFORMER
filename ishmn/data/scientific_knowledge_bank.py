"""
Scientific Knowledge Bank for Deep Neocortical Invariant Crystallization.
Comprehensive repository of physical, causal, thermodynamic, astronomical,
and biological ground-truth principles for UCE.
"""

from typing import List, Dict, Any

SCIENTIFIC_INVARIANTS = [
    # 1. Thermal Radiation & Wien's Displacement Law
    {
        "word": "wien",
        "domain": "physics_thermodynamics",
        "description": "Wien displacement law states peak thermal emission wavelength is inversely proportional to temperature.",
        "keywords": ["wien", "blackbody", "temperature", "wavelength", "peak"]
    },
    {
        "word": "radiation",
        "domain": "physics_thermodynamics",
        "description": "Thermal radiation emission shifts toward higher frequencies and blue light as temperature increases.",
        "keywords": ["radiation", "thermal", "emission", "spectrum", "heat"]
    },
    {
        "word": "blue",
        "domain": "physics_thermodynamics",
        "description": "Blue light possesses shorter wavelength and higher photon energy, indicating higher temperature than orange or red.",
        "keywords": ["blue", "hotter", "highest", "temperature", "energy"]
    },
    {
        "word": "temperature",
        "domain": "physics_thermodynamics",
        "description": "Order of temperature from highest to lowest for emitted colors is blue, then orange, then red.",
        "keywords": ["temperature", "hot", "cool", "thermal", "degrees"]
    },

    # 2. Mechanical Energy Conservation & Gravitational Invariants
    {
        "word": "conservation",
        "domain": "physics_mechanics",
        "description": "Total mechanical energy is conserved in free fall; kinetic energy gained equals potential energy lost.",
        "keywords": ["conservation", "energy", "constant", "mechanical", "vacuum"]
    },
    {
        "word": "potential",
        "domain": "physics_mechanics",
        "description": "Gravitational potential energy equals mass times gravity times height; it scales linearly with mass.",
        "keywords": ["potential", "gravitational", "height", "mass", "elevation"]
    },
    {
        "word": "kinetic",
        "domain": "physics_mechanics",
        "description": "Falling halfway down a height converts half of maximum potential energy into kinetic energy regardless of mass.",
        "keywords": ["kinetic", "motion", "velocity", "half", "maximum"]
    },
    {
        "word": "pendulum",
        "domain": "physics_mechanics",
        "description": "In a frictionless vacuum, a pendulum cannot swing higher than its release point due to conservation of energy.",
        "keywords": ["pendulum", "swing", "release", "amplitude", "frictionless"]
    },

    # 3. Atmospheric Thermodynamics & Greenhouse Physics
    {
        "word": "greenhouse",
        "domain": "atmospheric_physics",
        "description": "Water vapor is the most abundant and potent greenhouse gas in Earth's lower atmosphere.",
        "keywords": ["greenhouse", "gas", "vapor", "atmosphere", "warming"]
    },
    {
        "word": "vapor",
        "domain": "atmospheric_physics",
        "description": "Atmospheric water vapor accounts for the majority of natural greenhouse infrared absorption.",
        "keywords": ["vapor", "water", "humidity", "moisture", "troposphere"]
    },
    {
        "word": "diurnal",
        "domain": "atmospheric_physics",
        "description": "Solar radiation causes daytime diurnal warming, heating surface air from morning to afternoon peak around 75 to 80 Fahrenheit.",
        "keywords": ["diurnal", "daytime", "warming", "afternoon", "sunny"]
    },

    # 4. Cellular Biology & Structural Classification
    {
        "word": "prokaryote",
        "domain": "cellular_biology",
        "description": "Prokaryotic cells are distinguished by significantly smaller size (1 to 5 micrometers) and absence of a nucleus.",
        "keywords": ["prokaryote", "prokaryotic", "bacteria", "small", "size"]
    },
    {
        "word": "eukaryote",
        "domain": "cellular_biology",
        "description": "Eukaryotic cells are larger (10 to 100 micrometers) with membrane-bound organelles; both have plasma membranes.",
        "keywords": ["eukaryote", "eukaryotic", "nucleus", "organelles", "membrane"]
    },

    # 5. Celestial Mechanics & Rotational Dynamics
    {
        "word": "rotation",
        "domain": "celestial_mechanics",
        "description": "Increasing a planet's rotational speed shortens its planetary day length without affecting its orbital year.",
        "keywords": ["rotation", "spin", "day", "shorter", "period"]
    },
    {
        "word": "orbital",
        "domain": "celestial_mechanics",
        "description": "Planetary years depend solely on orbital distance from the star, not rotational spin rate.",
        "keywords": ["orbital", "year", "revolution", "distance", "orbit"]
    },

    # 6. Experimental Methodology & Variable Isolation
    {
        "word": "experiment",
        "domain": "experimental_methodology",
        "description": "Testing device longevity requires identical operating conditions and continuous operation until failure.",
        "keywords": ["experiment", "control", "variable", "longevity", "investigation"]
    },

    # 7. Ecological Dynamics & Predation
    {
        "word": "predation",
        "domain": "ecology",
        "description": "Predator presence exerts selective pressure and causes behavioral modification in prey populations.",
        "keywords": ["predation", "hawk", "chicken", "prey", "predator"]
    },

    # 8. Quantitative Reasoning & Profit Accounting
    {
        "word": "profit",
        "domain": "quantitative_reasoning",
        "description": "Net profit equals total revenue minus total investment and renovation expenses.",
        "keywords": ["profit", "revenue", "cost", "investment", "net"]
    },
    {
        "word": "percentage",
        "domain": "quantitative_reasoning",
        "description": "A percentage increase adds the fraction of original value to compute new total.",
        "keywords": ["percentage", "discount", "increase", "rate", "fraction"]
    },

    # 9. Cellular Bioenergetics & Respiration
    {
        "word": "mitochondria",
        "domain": "cellular_biology",
        "description": "Mitochondria are double-membraned organelles that synthesize adenosine triphosphate through cellular respiration and oxidative phosphorylation.",
        "keywords": ["mitochondria", "atp", "organelle", "respiration", "powerhouse"]
    },

    # 10. Fundamental Hydrocarbon Chemistry
    {
        "word": "methane",
        "domain": "chemistry",
        "description": "Methane is the simplest tetrahedral alkane consisting of one carbon and four hydrogen atoms, serving as the main component of fossil gas.",
        "keywords": ["methane", "hydrocarbon", "alkane", "gas", "tetrahedral"]
    },

    # 11. Computer Science Data Structures & Immutability
    {
        "word": "tuple",
        "domain": "computer_science",
        "description": "In Python, a tuple is an immutable ordered sequence, whereas a list is mutable and supports in-place modifications.",
        "keywords": ["tuple", "immutable", "sequence", "list", "ordered"]
    },

    # 12. Relational Database Theory
    {
        "word": "distinct",
        "domain": "databases",
        "description": "Relational query projection uses the distinct operator to suppress duplicate records from a multisets table output.",
        "keywords": ["distinct", "duplicate", "sql", "relational", "projection"]
    },

    # 13. Algorithmic Paradigms & Complexity
    {
        "word": "dynamic_programming",
        "domain": "algorithms",
        "description": "Dynamic programming is an optimization method combining optimal substructure with overlapping subproblems using memoization or tabulation.",
        "keywords": ["dynamic_programming", "memoization", "subproblems", "tabulation", "algorithm"]
    },
    {
        "word": "binary_search",
        "domain": "algorithms",
        "description": "Binary search divides a sorted collection in half at each step, achieving logarithmic search complexity.",
        "keywords": ["binary_search", "logarithmic", "sorted", "divide", "complexity"]
    }
]

