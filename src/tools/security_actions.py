"""
Mock security action functions that the core agent can trigger dynamically.

Each function logs the action via the standard logging module and returns a
structured confirmation string that the agent appends to its conversation
context for transparency and auditability.
"""

import logging

logger = logging.getLogger("edgeops.security_actions")


def lock_down_zone(zone_name: str) -> str:
    """Simulate locking down a physical security zone.

    In production this would interface with physical access-control systems
    (electronic locks, turnstiles, barriers). Here we log and return a
    confirmation string.

    Args:
        zone_name: Identifier of the zone to lock
                   (e.g. "kuzey_kapi", "dogu_cephe", "guney_mutfak").

    Returns:
        Confirmation message suitable for agent context injection.
    """
    logger.warning("🔒 LOCKDOWN initiated for zone: %s", zone_name)

    confirmation = (
        f"[GUVENLIK_AKSIYONU] lock_down_zone('{zone_name}') basarili. "
        f"{zone_name} bolgesindeki tum giris/cikis noktalari kilitlenmistir. "
        f"Kontrol merkezine bilgi verilmistir."
    )
    print(confirmation)
    return confirmation


def guvenli_bolge_uyarisi(zone_name: str, mesaj: str) -> str:
    """Send a safety warning to a specific zone (loudspeaker / SMS / push).

    Args:
        zone_name: Target zone identifier.
        mesaj: Warning message content in Turkish.

    Returns:
        Confirmation string.
    """
    logger.info("📢 Zone warning dispatched to %s: %s", zone_name, mesaj)

    confirmation = (
        f"[GUVENLIK_AKSIYONU] guvenli_bolge_uyarisi('{zone_name}') basarili. "
        f"Uyari mesaji: \"{mesaj}\""
    )
    print(confirmation)
    return confirmation
