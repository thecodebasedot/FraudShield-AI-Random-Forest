# Security & PCI-DSS Compliance

FraudShield AI implements the technical controls a payment-card environment is
expected to build on. This document maps project features to the relevant
**PCI-DSS v4.0** requirements.

> ⚠️ Implementing controls is **not** the same as being certified. A production
> deployment still needs an audited environment, network segmentation, a QSA
> assessment and organizational policies. This is the engineering foundation.

## Control mapping

| PCI-DSS area | Requirement (paraphrased) | How FraudShield addresses it |
|--------------|---------------------------|------------------------------|
| **Req 1 / 6** | Secure systems & network | Containerized, non-root pods, `readOnly`-friendly, least-privilege resource limits (`k8s/deployment.yaml`) |
| **Req 3** | Protect stored data | Field-level encryption at rest via Fernet (`src/security.py`, `FRAUDSHIELD_ENC_KEY`); Postgres over TLS via `DATABASE_URL` |
| **Req 3.4** | Mask PAN when displayed | `mask_pan` / `mask_sensitive` mask card numbers to last 4 and redact secrets before logging |
| **Req 4** | Encrypt data in transit | TLS-terminating ingress with forced HTTPS redirect (`k8s/ingress.yaml`) |
| **Req 7** | Restrict access by need-to-know | Multi-tenant isolation — each API key is scoped to a tenant; analytics never cross tenants |
| **Req 8** | Identify & authenticate access | API-key authentication, SHA-256 hashed at rest, revocable (`src/auth.py`) |
| **Req 8.3** | Strong secrets handling | Keys shown once, only hashes stored; runtime secrets via Kubernetes Secrets, never in the ConfigMap or repo |
| **Req 10** | Log & monitor all access | Append-only audit trail of every prediction & API action (`audit_logs` table) |
| **Req 6.4 / DoS** | Protect against abuse | Per-client token-bucket rate limiting + security response headers (`src/security.py`) |
| **Req 11** | Test security regularly | CI runs the test suite (incl. auth, tenant isolation, rate limiting) on every push |

## Operational checklist for a real deployment

- [ ] Set `FRAUDSHIELD_ENC_KEY` from a managed secret store (not env files).
- [ ] Use PostgreSQL over TLS; enable encryption at rest at the DB layer too.
- [ ] Terminate TLS at the ingress; disable plain HTTP.
- [ ] Rotate API keys and the encryption key on a schedule.
- [ ] Ship audit logs to a tamper-evident, centralized store (SIEM).
- [ ] Run in a segmented network; restrict egress.
- [ ] Never log full PANs — `mask_sensitive` is the last line, not the only one.

## Quick verification

```bash
# Generate an encryption key
python -m src.security genkey

# Confirm masking
python -c "from src.security import mask_pan; print(mask_pan('card 4111 1111 1111 1111 used'))"
# -> card ************1111 used
```
