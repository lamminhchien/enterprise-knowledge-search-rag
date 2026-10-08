# Enterprise IT Security Standard Operating Procedure (SOP-SEC-2026)

## 1. Authentication and Access Control
All employees accessing enterprise infrastructure (including Microsoft 365, internal VPN, and code repositories) must enforce hardware-backed Multi-Factor Authentication (MFA). 
- Passwords must be at least 16 characters long and changed every 90 days if compromise is suspected.
- Privileged access (Admin roles) requires explicit approval from the Chief Information Security Officer (CISO) and follows the Principle of Least Privilege (PoLP).

## 2. Session Management and Auto-Lockout
- Idle sessions on workstation clients must automatically lock after 5 minutes of inactivity.
- Any unauthorized physical tampering or 5 consecutive failed authentication attempts will lock the account and alert the Security Operations Center (SOC).

## 3. Incident Escalation Protocol
In the event of suspected credential compromise or data leakage:
- Immediate isolation of affected network subnets within 15 minutes.
- Notification to the Data Protection Officer (DPO) and emergency escalation hotline: `sec-ops@corp-internal.com`.
- Mandatory forensic log retention for 365 days.
