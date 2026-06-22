"""
Mock emergency response functions for the EdgeOps security agent.

These simulate coordination with civil defence / military search-and-rescue
teams. In production they would integrate with national emergency dispatch
APIs or radio gateways.
"""

import logging

logger = logging.getLogger("edgeops.emergency_actions")


def call_emergency_services(location: str, tactical_priority: str) -> str:
    """Simulate placing an emergency call to rapid-response teams.

    Args:
        location: Geographic or zone-level description
                  (e.g. "dogu_cephe, grid_3").
        tactical_priority:   One of "dusuk", "orta", "yuksek", "kritik".
                             Determines response team type and ETA.

    Returns:
        Confirmation string with dispatch details.
    """
    priority_map = {
        "kritik": {
            "ekipler": "Ozel Harekat + 112 Acil Tip",
            "eta": "2 dakika",
        },
        "yuksek": {
            "ekipler": "Guvenlik Mudahale + 112 Acil Tip",
            "eta": "5 dakika",
        },
        "orta": {
            "ekipler": "Guvenlik Devriye",
            "eta": "10 dakika",
        },
        "dusuk": {
            "ekipler": "Bilgi Amacli - Standart Devriye",
            "eta": "30 dakika",
        },
    }

    details = priority_map.get(tactical_priority, priority_map["dusuk"])

    logger.critical(
        "🚨 EMERGENCY DISPATCH | Location: %s | Priority: %s | Teams: %s | ETA: %s",
        location,
        tactical_priority,
        details["ekipler"],
        details["eta"],
    )

    confirmation = (
        f"[ACIL_DURUM] call_emergency_services('{location}', "
        f"'{tactical_priority}') basarili.\n"
        f"  |  Koordinat / Bolge: {location}\n"
        f"  |  Gorevlendirilen Ekipler: {details['ekipler']}\n"
        f"  |  Tahmini Varis Suresi: {details['eta']}\n"
        f"  |  Oncelik Seviyesi: {tactical_priority}"
    )
    print(confirmation)
    return confirmation


def bilgi_amacli_bildirim(baslik: str, icerik: str) -> str:
    """Send an informational notification to the command centre.

    This is a non-urgent action used for low-risk events that still
    require logging and human awareness.

    Args:
        baslik: Short notification title.
        icerik: Detailed message body.

    Returns:
        Confirmation string.
    """
    logger.info("📋 INFO NOTIFICATION | %s: %s", baslik, icerik)

    confirmation = (
        f"[ACIL_DURUM] bilgi_amacli_bildirim('{baslik}') basarili.\n"
        f"  |  Icerik: {icerik}\n"
        f"  |  Komuta merkezine iletilmistir."
    )
    print(confirmation)
    return confirmation
