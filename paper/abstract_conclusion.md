# Abstract & Conclusion (DRAFT)

> The two "bookend" pieces, written for the paper (reviewers / co-author). The Abstract goes in the
> frontmatter; the Conclusion is §6. **[exfil]** marks clauses that depend on including the data-
> exfiltration second attack family and trim cleanly if scoped out. Abstract is ≤250 words (COSE rule).

## Abstract

Machine-learning network intrusion detectors are usually evaluated for accuracy and, when attacked, in
the *feature space* — by perturbing the flow feature vector directly, a perturbation that may
correspond to no traffic an attacker could actually send. We study the harder, operationally relevant
question of *problem-space* robustness: resistance to perturbations that are physically realizable as
traffic and preserve the attack's function. We build a model-in-the-loop evasion that generates real
attack traffic in an emulated network, extracts its flow through the detector's own pipeline, and
admits a perturbation only when a functional oracle confirms the attack still works. Using command-and-
control beaconing — whose detectable timing regularity is an attacker-controlled, functionally-free
parameter — with a realistic benign mix that keeps clean detection accurate but not saturated, we find
a flow-level detector is evaded for 100% of instances by a trivial change in timing jitter. Comparing a
classical neural network against a quantum-classical variational classifier on matched inputs, we find
*no quantum robustness advantage*: an apparent edge on a saturated task is an artifact of trivial
separability and vanishes once detection is realistic. Adversarial training does not restore
robustness — it drives one axis's evasion to zero at a false-positive cost, but the vulnerability
*relocates* to the next functionally-free axis, which re-breaks the hardened detector. **[exfil]** The
same conclusions hold for a second, structurally different attack family (data exfiltration). Flow-
level learned detection of evasion-amenable attacks is thus fragile under realizable, functionality-
preserving evasion, and quantum models offer no inherent remedy.

> [TODO] Trim to COSE's 250-word limit after the final [exfil] scope decision (the [exfil] sentence is
> the natural cut if exfil is excluded). Current length ≈ 235 words with the [exfil] sentence.

## 6. Conclusion

We studied the problem-space adversarial robustness of flow-level machine-learning intrusion detectors
— their resistance to realizable, functionality-preserving traffic manipulation rather than to feature-
vector perturbations. Driving a model-in-the-loop evasion against real traffic with a functional
oracle, we found that a realistic detector is evaded for 100% of instances at negligible cost, that a
quantum-classical classifier offers no robustness advantage over a classical one (and that an apparent
advantage is a saturation artifact of a too-easy task), and that single-axis adversarial training
relocates the vulnerability rather than closing it, at a false-positive cost. **[exfil]** These
conclusions held across two structurally different attack families, indicating the fragility is
intrinsic to flow-level learned detection of evasion-amenable attacks rather than specific to one
attack. The binding constraint for the defender is not the attacker's effort — which is trivial — but
the defender's own false-positive budget: a durable defense would have to cover the joint space of
functionally-free realizable axes without flagging the benign traffic those axes lead into. Whether
that is achievable is, we believe, the central open question these results pose.
