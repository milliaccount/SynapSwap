#  Contributing to SynapSwap

Thank you for your interest in **SynapSwap**! We welcome and appreciate contributions of all kinds — whether you are fixing bugs, improving performance, refining the architecture, or introducing new features.

By contributing, you help make advanced AI inference **more accessible on consumer-grade hardware**.

---

##  How to Contribute

### 1. Report a Bug or Propose an Idea

Before submitting any code, please consider opening an **Issue** to:

* Report unexpected behavior, crashes, or regressions
* Propose a new feature (e.g., support for an additional backend)
* Discuss architectural or performance improvements

Clear, well-documented issues help ensure productive discussions and efficient development.

---

### 2. Submit a Pull Request (PR)

1. **Fork** the repository
2. Create a **topic branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Implement your changes
4. Ensure the project builds cleanly without warnings:

   ```bash
   make clean && make
   ```
5. **Commit** your changes with clear, descriptive messages
6. **Push** your branch to your fork and open a **Pull Request**

Each Pull Request should clearly explain the motivation, scope, and impact of the changes.

---

##  Coding Standards (Style Guide)

To keep the codebase clean, portable, and maintainable, SynapSwap follows these conventions:

* **Language**: Strict C99 (for maximum portability)
* **Naming Conventions**:

  * Public API functions: `synapswap_function_name()`
  * Internal functions: `internal_function_name()` or `_function_name()`
  * Variables: `snake_case` (e.g., `vram_limit`)
* **Documentation**:

  * Use **Doxygen-style comments** (`/** ... */`) for all public functions declared in `.h` files
* **Memory Management**:

  * Every `synapswap_malloc()` **must** have a corresponding `synapswap_free()`
  * Memory leaks are not tolerated

---

##  Code Architecture Overview (for Developers)

If you want to dive into the internals, the following reading order is recommended:

1. `include/synapswap.h`
   Public API exposed to users and external runtimes

2. `src/core/intercept.c`
   Entry point responsible for intercepting and managing memory allocations

3. `src/core/scheduler.c`
   Graph-aware logic that determines **what** to swap, **when**, and **why**

4. `src/core/transfer_engine.c`
   Background thread and PCIe transfer management (asynchronous engine)

---

##  Testing

Any major feature or architectural change **must** be accompanied by tests located in the `tests/` directory.

Before submitting your Pull Request, please run the stress tests to validate stability and correctness:

```bash
make
./synapswap_test
```

---

##  Code of Conduct

Please be respectful, constructive, and supportive of fellow contributors. We value a collaborative environment where newcomers feel welcome and experienced developers can exchange ideas openly.

Harassment, hostility, or disrespectful behavior will not be tolerated.

---

Thank you again for helping improve SynapSwap.

**DamienOS**
