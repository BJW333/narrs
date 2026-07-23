"""
Optional domain plugins.

Core NARRS never imports anything from this package: importing an adapter is a
deliberate choice by the caller, and the optimizer has no knowledge of it. That
one-way dependency is what keeps the core domain-free.

    from narrs.adapters.trading import CostModel, cost_stress_contexts
"""
