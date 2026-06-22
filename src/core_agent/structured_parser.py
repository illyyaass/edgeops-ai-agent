"""
Pydantic v2 models (KararDestekRaporu, OlayDetayi) and a helper function
that configures vLLM's GuidedDecodingParams with the Pydantic schema to
enforce strict JSON compliance via FSM-guided decoding.

By passing 'guided_decoding' to the vLLM sampling parameters, the engine
will constrain token generation to only produce valid JSON that matches
the provided Pydantic model, eliminating markdown blocks and formatting
errors at the token level.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────
# Enums for constrained string fields
# ──────────────────────────────────────────────────────────────────────


class OlayTuru(str, Enum):
    """Constrained set of event types the agent can detect."""

    intrusion = "intrusion"
    theft = "theft"
    panic = "panic"
    fire = "fire"
    suspicious_person = "suspicious_person"
    abandoned_object = "abandoned_object"
    normal = "normal"
    other = "other"


class GridPosition(str, Enum):
    """5-grid split-screen position identifiers."""

    grid_1 = "grid_1"
    grid_2 = "grid_2"
    grid_3 = "grid_3"
    grid_4 = "grid_4"
    grid_5 = "grid_5"


class ZoneName(str, Enum):
    """Geographic zone names for the surveillance layout."""

    kuzey_kapi = "kuzey_kapi"
    dogu_cephe = "dogu_cephe"
    guney_mutfak = "guney_mutfak"
    bati_merdiven = "bati_merdiven"
    merkez_hol = "merkez_hol"
    belirtilmemis = "belirtilmemis"


class RiskSeviyesi(str, Enum):
    """Overall risk level for the scene."""

    dusuk = "dusuk"
    orta = "orta"
    yuksek = "yuksek"
    kritik = "kritik"


class EylemFonksiyonu(str, Enum):
    """Available tool functions the agent can invoke."""

    lock_down_zone = "lock_down_zone"
    call_emergency_services = "call_emergency_services"
    guvenli_bolge_uyarisi = "guvenli_bolge_uyarisi"
    bilgi_amacli = "bilgi_amacli"


# ──────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────


class OlayDetayi(BaseModel):
    """A single detected event within one grid cell."""

    event_id: int = Field(
        ...,
        ge=1,
        le=5,
        description="Benzersiz olay numarasi (1-tabanli, artan sira ile).",
    )
    tur: OlayTuru = Field(
        ...,
        description="Olayin kategorik turu. Enum disi deger kabul edilmez.",
    )
    aciklama: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Olayin detayli Turkce aciklamasi.",
    )
    grid_position: GridPosition = Field(
        ...,
        description="Olayin gerceklestigi grid penceresi.",
    )
    zone: ZoneName = Field(
        default=ZoneName.belirtilmemis,
        description="Olayin gerceklestigi cografi bolge / sektor.",
    )
    baslangic_zamani: str = Field(
        ...,
        description=(
            "Baslangic zamani. Format: 'ss:dd:dd.xxx' veya 'frame_index:N'."
        ),
    )
    bitis_zamani: str = Field(
        ...,
        description=(
            "Bitis zamani. Devam ediyorsa 'devam_ediyor' yaz."
        ),
    )
    seviye: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Guven skoru / tespit seviyesi (0.0 - 1.0).",
    )

    @field_validator("bitis_zamani")
    @classmethod
    def _validate_bitis(cls, v: str) -> str:
        """Allow either a timestamp-like string or 'devam_ediyor'."""
        allowed = {"devam_ediyor"}
        if v in allowed:
            return v
        if ":" in v or v.startswith("frame_index:"):
            return v
        raise ValueError(
            f"'{v}' gecersiz. 'ss:dd:dd.xxx', 'frame_index:N' veya "
            f"'devam_ediyor' olmalidir."
        )


class RiskAnalizi(BaseModel):
    """Aggregated risk assessment across all grid cells."""

    genel_risk_seviyesi: RiskSeviyesi = Field(
        ...,
        description="Genel risk seviyesi (dusuk/orta/yuksek/kritik).",
    )
    en_tehlikeli_grid: GridPosition = Field(
        ...,
        description="En yuksek riskli grid penceresi.",
    )
    aciklama: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Risk degerlendirme aciklamasi.",
    )


class EylemOnerisi(BaseModel):
    """A single recommended action for the human operator or automation."""

    fonksiyon: EylemFonksiyonu = Field(
        ...,
        description="Cagrilacak arac fonksiyonu.",
    )
    parametreler: dict[str, Any] = Field(
        default_factory=dict,
        description="Fonksiyona gonderilecek anahtar-deger parametreleri.",
    )
    gerekce: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Bu eylemin neden secildigine dair gerekce.",
    )


class KararDestekRaporu(BaseModel):
    """Top-level decision-support report produced by the agent.

    This is the single validated JSON structure that the vLLM engine will
    be forced to generate via GuidedDecoding / FSM constraints.
    """

    summary: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Olayin kisa Turkce ozeti (en fazla 3 cumle).",
    )
    events: list[OlayDetayi] = Field(
        ...,
        min_length=0,
        max_length=5,
        description="Tespit edilen olaylar listesi (en fazla 5).",
    )
    risk_analizi: RiskAnalizi = Field(
        ...,
        description="Risk degerlendirme bilgisi.",
    )
    eylem_onerileri: list[EylemOnerisi] = Field(
        default_factory=list,
        description="Onerilen aksiyonlar listesi (bos olabilir).",
    )

    @field_validator("events")
    @classmethod
    def _benzersiz_event_id(cls, events: list[OlayDetayi]) -> list[OlayDetayi]:
        """Ensure all event_id values are unique."""
        ids = [e.event_id for e in events]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id degerleri benzersiz olmalidir.")
        return events


# ──────────────────────────────────────────────────────────────────────
# vLLM GuidedDecodingParams factory
# ──────────────────────────────────────────────────────────────────────


def build_guided_decoding_params(
    model: type[BaseModel] = KararDestekRaporu,
    /,
) -> dict[str, Any]:
    """Return a dict suitable for vLLM's ``guided_decoding`` parameter.

    When passed to ``SamplingParams`` via ``guided_decoding=…``, the vLLM
    engine uses the Pydantic model's JSON schema to build a finite-state
    machine (FSM) that constrains every generated token, guaranteeing
    valid JSON that matches the Pydantic model — no markdown fences,
    no extra text, and no partial or malformed responses.

    Usage:
        >>> from vllm import SamplingParams
        >>> params = SamplingParams(
        ...     temperature=0.1,
        ...     max_tokens=2048,
        ...     guided_decoding=build_guided_decoding_params(),
        ... )

    Returns:
        Dictionary with ``"json_schema"`` or ``"guided_json"`` key as
        expected by vLLM's ``GuidedDecodingParams``. (The exact key
        depends on the vLLM version; we provide both for compatibility.)
    """
    schema = model.model_json_schema()

    return {
        "json_schema": schema,
        "guided_json": schema,
    }


def parse_rapor(json_str: str) -> KararDestekRaporu:
    """Deserialize and validate a raw JSON string into a KararDestekRaporu.

    This is a safety net for scenarios where GuidedDecoding is not active
    (e.g. local testing with the raw model). In production with guided
    decoding, this should never fail unless the vLLM engine is bypassed.

    Args:
        json_str: Raw JSON string from the model output.

    Returns:
        Validated ``KararDestekRaporu`` instance.

    Raises:
        pydantic.ValidationError: If the JSON does not match the schema.
    """
    import json

    data = json.loads(json_str)
    return KararDestekRaporu.model_validate(data)
