"""Map Go/No-Go gates plus data inventory to a journal recommendation."""

from __future__ import annotations


def recommend_journal(gates: dict, inventory: dict) -> dict:
    has_true_sif = bool(inventory.get("gosif") or inventory.get("tropomi_sif"))
    has_vod = bool(inventory.get("vodca") or inventory.get("smap_vod"))
    has_trade = bool(inventory.get("bilateral_trade"))
    has_optical = bool(inventory.get("modis_optical"))
    phys = gates.get("physiology_lead", {})
    trade = gates.get("trade_value", {})
    pheno = gates.get("dynamic_phenology", {})
    unc = gates.get("uncertainty", {})
    notes = []
    if not has_true_sif:
        notes.append("Photosynthesis channel is MOD17 GPP, not GOSIF/TROPOMI; H1 cannot be claimed as an SIF result.")
    elif not phys.get("go"):
        notes.append("GOSIF is on the table, but it does not add reliable lead beyond optical+climate.")
    if not has_vod:
        notes.append("No VOD product; B5 / canopy-water increment is untested.")
    elif inventory.get("smap_vod") and not inventory.get("vodca"):
        notes.append("Canopy-water channel is SMAP vegetation water content, not VODCA; treat 2015+ as a sub-experiment.")
    if not has_trade:
        notes.append("No bilateral trade matrix; Nature Food food-system hook is incomplete.")
    if inventory.get("headline_crops"):
        extras = []
        if inventory.get("seasonal_rice") or inventory.get("seasonal_rice_held_out"):
            extras.append(
                "FAO rice held out. India kharif/Aman was dropped (canopy ≠ APY). "
                "China NBS provincial early rice (90–195) couples to RS "
                "(harvest R²≈0.61, 2020 Yangtze flood visible) but B4 adds no lead "
                "beyond optical, so it stays out of the headline gate sample."
            )
        else:
            extras.append("FAO rice is a multi-season annual total and is held out of the yield model.")
        if inventory.get("usa_maize_held_out") or inventory.get("usa_maize"):
            extras.append(
                "US NASS maize (county and CROPGRIDS-weighted state) couples in-sample "
                "(July SIF r≈0.39) but rolling-origin harvest R² is negative."
            )
        if inventory.get("eu_maize_held_out") or inventory.get("eu_maize"):
            extras.append(
                "Eurostat NUTS maize has harvest R²≈0.07 and no earliest_lead."
            )
        notes.append("Headline sample is " + ",".join(inventory["headline_crops"]) + ". " + " ".join(extras))

    if has_true_sif and has_trade and phys.get("go") and trade.get("go"):
        target = "Nature Food"
        reason = "Earlier physiological warning plus a quantitative trade-exposure gain are both present."
    elif has_optical and (phys.get("go") or pheno.get("go")) and not (has_trade and trade.get("go")):
        target = "Remote Sensing of Environment"
        reason = "Remote-sensing increment is the supported claim; the food-system module is too weak for Nature Food."
    elif has_trade and trade.get("go") and not phys.get("go"):
        target = "Nature Communications / RSE"
        reason = "Trade exposure can be shown, but the extra lead time is not a physiological early-warning result."
    elif pheno.get("go") and not phys.get("go"):
        target = "Remote Sensing of Environment / CEE"
        reason = "Dynamic-phenology bias is the cleanest result; keep it as a phenology paper."
    else:
        target = "Remote Sensing of Environment"
        reason = "Current observational evidence does not clear the Nature Food Go gates."

    shock = gates.get("shock_year_physiology", {})
    if shock.get("n"):
        n_cy = shock.get("n_country_crop_years")
        n_note = f"n_pred={shock.get('n')}"
        if n_cy:
            n_note += f", n_country-crop-years={n_cy}, n_countries={shock.get('n_countries')}"
        notes.append(
            f"Event-year / |z|>=1 physiology lead gain is {shock.get('value')} d "
            f"(B4 {shock.get('b4_lead')} vs B2 {shock.get('b2_lead')}, {n_note}). "
            "n_pred counts lead×country-crop-year rows, not independent samples."
        )
    if not unc.get("go"):
        notes.append("Do not put 'trustworthy' in the title until 90% coverage is calibrated.")

    return {
        "recommendation": target,
        "reason": reason,
        "gates": gates,
        "inventory": inventory,
        "notes": notes,
    }
