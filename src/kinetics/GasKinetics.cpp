/**
 *  @file GasKinetics.cpp Homogeneous kinetics in ideal gases
 */

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/kinetics/GasKinetics.h"
#include "cantera/thermo/ThermoPhase.h"

using namespace std;

namespace Cantera
{
GasKinetics::GasKinetics(ThermoPhase* thermo) :
    BulkKinetics(thermo),
    m_logp_ref(0.0),
    m_logc_ref(0.0),
    m_logStandConc(0.0),
    m_pres(0.0)
{
    setJacobianSettings(AnyMap()); // use default settings
}

void GasKinetics::finalizeSetup()
{
    // Third-body calculators
    m_3b_concm.finalizeSetup(m_kk, nReactions());
    m_falloff_concm.finalizeSetup(m_kk, nReactions());

    m_rbuf0.resize(nReactions());
    m_rbuf1.resize(nReactions());
    m_rbuf2.resize(nReactions());

    BulkKinetics::finalizeSetup();
}

void GasKinetics::update_rates_T()
{
    double T = thermo().temperature();
    double P = thermo().pressure();
    m_logStandConc = log(thermo().standardConcentration());
    double logT = log(T);

    if (T != m_temp) {
        if (!m_rfn.empty()) {
            m_rates.update(T, logT, m_rfn.data());
        }

        if (!m_rfn_low.empty()) {
            m_falloff_low_rates.update(T, logT, m_rfn_low.data());
            m_falloff_high_rates.update(T, logT, m_rfn_high.data());
        }
        if (!falloff_work.empty()) {
            m_falloffn.updateTemp(T, falloff_work.data());
        }
        updateKc();
        m_ROP_ok = false;
        if (m_blowersmasel_rates.nReactions()) {
            thermo().getPartialMolarEnthalpies(m_grt.data());
            getReactionDelta(m_grt.data(), m_dH.data());
            m_blowersmasel_rates.updateBlowersMasel(T, logT, m_rfn.data(), m_dH.data());
        }
    }

    if (T != m_temp || P != m_pres) {

        // loop over MultiBulkRates evaluators
        for (auto& rates : m_bulk_rates) {
            rates->update(thermo(), m_concm.data());
            rates->getRateConstants(thermo(), m_rfn.data(), m_concm.data());
        }

        if (m_plog_rates.nReactions()) {
            m_plog_rates.update(T, logT, m_rfn.data());
            m_ROP_ok = false;
        }

        if (m_cheb_rates.nReactions()) {
            m_cheb_rates.update(T, logT, m_rfn.data());
            m_ROP_ok = false;
        }
    }
    m_pres = P;
    m_temp = T;
}

void GasKinetics::update_rates_C()
{
    thermo().getActivityConcentrations(m_act_conc.data());
    thermo().getConcentrations(m_phys_conc.data());
    doublereal ctot = thermo().molarDensity();

    // 3-body reactions
    if (!concm_3b_values.empty()) {
        m_3b_concm.update(m_phys_conc, ctot, concm_3b_values.data());
    }

    // Falloff reactions
    if (!concm_falloff_values.empty()) {
        m_falloff_concm.update(m_phys_conc, ctot, concm_falloff_values.data());
    }

    // Third-body objects interacting with MultiRate evaluator
    if (!concm_multi_values.empty()) {
        // using pre-existing third-body handlers requires copying;
        m_multi_concm.update(m_phys_conc, ctot, concm_multi_values.data());
        for (size_t i = 0; i < m_multi_indices.size(); i++) {
            m_concm[m_multi_indices[i]] = concm_multi_values[i];
        }
    }

    // P-log reactions
    if (m_plog_rates.nReactions()) {
        double logP = log(thermo().pressure());
        m_plog_rates.update_C(&logP);
    }

    // Chebyshev reactions
    if (m_cheb_rates.nReactions()) {
        double log10P = log10(thermo().pressure());
        m_cheb_rates.update_C(&log10P);
    }

    m_ROP_ok = false;
}

void GasKinetics::updateKc()
{
    thermo().getStandardChemPotentials(m_grt.data());
    fill(m_rkcn.begin(), m_rkcn.end(), 0.0);

    // compute Delta G^0 for all reversible reactions
    getRevReactionDelta(m_grt.data(), m_rkcn.data());

    doublereal rrt = 1.0 / thermo().RT();
    for (size_t i = 0; i < m_revindex.size(); i++) {
        size_t irxn = m_revindex[i];
        m_rkcn[irxn] = std::min(exp(m_rkcn[irxn]*rrt - m_dn[irxn]*m_logStandConc),
                                BigNumber);
    }

    for (size_t i = 0; i != m_irrev.size(); ++i) {
        m_rkcn[ m_irrev[i] ] = 0.0;
    }
}

void GasKinetics::getEquilibriumConstants(doublereal* kc)
{
    update_rates_T();
    thermo().getStandardChemPotentials(m_grt.data());
    fill(m_rkcn.begin(), m_rkcn.end(), 0.0);

    // compute Delta G^0 for all reactions
    getReactionDelta(m_grt.data(), m_rkcn.data());

    doublereal rrt = 1.0 / thermo().RT();
    for (size_t i = 0; i < nReactions(); i++) {
        kc[i] = exp(-m_rkcn[i]*rrt + m_dn[i]*m_logStandConc);
    }

    // force an update of T-dependent properties, so that m_rkcn will
    // be updated before it is used next.
    m_temp = 0.0;
}

void GasKinetics::processFalloffReactions(double* ropf)
{
    // use m_ropr for temporary storage of reduced pressure
    vector_fp& pr = m_ropr;

    for (size_t i = 0; i < m_falloff_low_rates.nReactions(); i++) {
        pr[i] = concm_falloff_values[i] * m_rfn_low[i] / (m_rfn_high[i] + SmallNumber);
        AssertFinite(pr[i], "GasKinetics::processFalloffReactions",
                     "pr[{}] is not finite.", i);
    }

    m_falloffn.pr_to_falloff(pr.data(), falloff_work.data());

    for (size_t i = 0; i < m_falloff_low_rates.nReactions(); i++) {
        if (reactionTypeStr(m_fallindx[i]) == "falloff") {
            pr[i] *= m_rfn_high[i];
        } else { // CHEMACT_RXN
            pr[i] *= m_rfn_low[i];
        }
        ropf[m_fallindx[i]] = pr[i];
    }
}

void GasKinetics::updateROP()
{
    processFwdRateCoefficients(m_ropf.data());
    processThirdBodies(m_ropf.data());
    copy(m_ropf.begin(), m_ropf.end(), m_ropr.begin());

    // multiply ropf by concentration products
    m_reactantStoich.multiply(m_act_conc.data(), m_ropf.data());

    // for reversible reactions, multiply ropr by concentration products
    processEquilibriumConstants(m_ropr.data());
    m_revProductStoich.multiply(m_act_conc.data(), m_ropr.data());
    for (size_t j = 0; j != nReactions(); ++j) {
        m_ropnet[j] = m_ropf[j] - m_ropr[j];
    }

    for (size_t i = 0; i < m_rfn.size(); i++) {
        AssertFinite(m_rfn[i], "GasKinetics::updateROP",
                     "m_rfn[{}] is not finite.", i);
        AssertFinite(m_ropf[i], "GasKinetics::updateROP",
                     "m_ropf[{}] is not finite.", i);
        AssertFinite(m_ropr[i], "GasKinetics::updateROP",
                     "m_ropr[{}] is not finite.", i);
    }
    m_ROP_ok = true;
}

void GasKinetics::getFwdRateConstants(double* kfwd)
{
    processFwdRateCoefficients(m_ropf.data());

    if (legacy_rate_constants_used()) {
        warn_deprecated("GasKinetics::getFwdRateConstants",
            "Behavior to change after Cantera 2.6;\nresults will no longer include "
            "third-body concentrations for three-body reactions.\nTo switch to new "
            "behavior, use 'cantera.use_legacy_rate_constants(False)' (Python),\n"
            "'useLegacyRateConstants(0)' (MATLAB), 'Cantera::use_legacy_rate_constants"
            "(false)' (C++),\nor 'ct_use_legacy_rate_constants(0)' (clib).");

        processThirdBodies(m_ropf.data());
    }

    // copy result
    copy(m_ropf.begin(), m_ropf.end(), kfwd);
}


void GasKinetics::getJacobianSettings(AnyMap& settings) const
{
    settings["constant-pressure"] = m_jac_const_pressure;
    settings["mole-fractions"] = m_jac_mole_fractions;
    settings["exact-temperature-derivatives"] = m_jac_exact_ddT;
    settings["skip-third-bodies"] = m_jac_skip_third_bodies;
    settings["skip-falloff"] = m_jac_skip_falloff;
    settings["atol-delta-T"] = m_jac_atol_deltaT;
}

void GasKinetics::setJacobianSettings(const AnyMap& settings)
{
    m_jac_const_pressure = settings.getBool("constant-pressure", true);
    m_jac_mole_fractions = settings.getBool("mole-fractions", false);
    m_jac_exact_ddT = settings.getBool("exact-temperature-derivatives", false);
    m_jac_skip_third_bodies = settings.getBool("skip-third-bodies", false);
    m_jac_skip_falloff = settings.getBool("skip-falloff", true);
    m_jac_atol_deltaT = settings.getDouble("atol-delta-T", 1e-6);
}

void GasKinetics::scaleConcentrations(double *rates)
{
    double ctot = thermo().molarDensity();
    for (size_t i = 0; i < nReactions(); ++i) {
        rates[i] *= ctot;
    }
}

Eigen::SparseMatrix<double> GasKinetics::fwdRatesOfProgress_ddC()
{
    Eigen::SparseMatrix<double> jac;
    vector_fp& rop_rates = m_rbuf0;
    vector_fp& rop_stoich = m_rbuf1;
    vector_fp& rop_3b = m_rbuf2;

    // forward reaction rate coefficients
    processFwdRateCoefficients(rop_rates.data());
    if (m_jac_mole_fractions) {
        scaleConcentrations(rop_rates.data());
    }

    // derivatives handled by StoichManagerN
    copy(rop_rates.begin(), rop_rates.end(), rop_stoich.begin());
    processThirdBodies(rop_stoich.data());
    jac = m_reactantStoich.speciesDerivatives(m_act_conc.data(), rop_stoich.data());

    // derivatives handled by ThirdBodyCalc
    if (!m_jac_skip_third_bodies && !concm_multi_values.empty()) {
        if (!concm_3b_values.empty()) {
            // Do not support legacy CTI/XML-based reaction rate evaluators
            throw CanteraError("GasKinetics::fwdRatesOfProgress_ddC",
                "Not supported for legacy input format.");
        }
        copy(rop_rates.begin(), rop_rates.end(), rop_3b.begin());
        m_reactantStoich.multiply(m_act_conc.data(), rop_3b.data());
        jac += m_multi_concm.speciesDerivatives(rop_3b.data());
    }

    return jac;
}

Eigen::SparseMatrix<double> GasKinetics::revRatesOfProgress_ddC()
{
    Eigen::SparseMatrix<double> jac;
    vector_fp& rop_rates = m_rbuf0;
    vector_fp& rop_stoich = m_rbuf1;
    vector_fp& rop_3b = m_rbuf2;

    // reverse reaction rate coefficients
    processFwdRateCoefficients(rop_rates.data());
    processEquilibriumConstants(rop_rates.data());
    if (m_jac_mole_fractions) {
        scaleConcentrations(rop_rates.data());
    }

    // derivatives handled by StoichManagerN
    copy(rop_rates.begin(), rop_rates.end(), rop_stoich.begin());
    processThirdBodies(rop_stoich.data());
    jac = m_revProductStoich.speciesDerivatives(m_act_conc.data(), rop_stoich.data());

    // derivatives handled by ThirdBodyCalc
    if (!m_jac_skip_third_bodies && !concm_multi_values.empty()) {
        if (!concm_3b_values.empty()) {
            // Do not support legacy CTI/XML-based reaction rate evaluators
            throw CanteraError("GasKinetics::revRatesOfProgress_ddC",
                "Not supported for legacy input format.");
        }
        copy(rop_rates.begin(), rop_rates.end(), rop_3b.begin());
        m_revProductStoich.multiply(m_act_conc.data(), rop_3b.data());
        jac += m_multi_concm.speciesDerivatives(rop_3b.data());
    }

    return jac;
}

Eigen::SparseMatrix<double> GasKinetics::netRatesOfProgress_ddC()
{
    Eigen::SparseMatrix<double> jac;
    vector_fp& rop_rates = m_rbuf0;
    vector_fp& rop_stoich = m_rbuf1;
    vector_fp& rop_3b = m_rbuf2;

    // forward reaction rate coefficients
    processFwdRateCoefficients(rop_rates.data());
    if (m_jac_mole_fractions) {
        scaleConcentrations(rop_rates.data());
    }
    copy(rop_rates.begin(), rop_rates.end(), rop_stoich.begin());

    // forward derivatives handled by StoichManagerN
    processThirdBodies(rop_stoich.data());
    jac = m_reactantStoich.speciesDerivatives(
        m_act_conc.data(), rop_stoich.data());

    // forward derivatives handled by ThirdBodyCalc
    if (!m_jac_skip_third_bodies && !concm_multi_values.empty()) {
        if (!concm_3b_values.empty()) {
            // Do not support legacy CTI/XML-based reaction rate evaluators
            throw CanteraError("GasKinetics::revRatesOfProgress_ddC",
                "Not supported for legacy input format.");
        }
        copy(rop_rates.begin(), rop_rates.end(), rop_3b.begin());
        m_reactantStoich.multiply(m_act_conc.data(), rop_3b.data());
        jac += m_multi_concm.speciesDerivatives(rop_3b.data());
    }

    // reverse reaction rate coefficients
    processEquilibriumConstants(rop_rates.data());
    copy(rop_rates.begin(), rop_rates.end(), rop_stoich.begin());

    // reverse derivatives handled by StoichManagerN
    processThirdBodies(rop_stoich.data());
    jac -= m_revProductStoich.speciesDerivatives(
        m_act_conc.data(), rop_stoich.data());

    // reverse derivatives handled by ThirdBodyCalc
    if (!m_jac_skip_third_bodies && !concm_multi_values.empty()) {
        copy(rop_rates.begin(), rop_rates.end(), rop_3b.begin());
        m_revProductStoich.multiply(m_act_conc.data(), rop_3b.data());
        jac -= m_multi_concm.speciesDerivatives(rop_3b.data());
    }

    return jac;
}

Eigen::VectorXd GasKinetics::ratesOfProgress_ddT_constP(bool forward, bool reverse)
{
    Eigen::VectorXd out(nReactions());
    double dTinv = 1. / m_jac_atol_deltaT;
    double T = thermo().temperature();
    double P = thermo().pressure();

    out.fill(0.);
    thermo().setState_TP(T + m_jac_atol_deltaT, P);
    updateROP();
    if (forward && reverse) {
        out += MappedVector(m_ropnet.data(), m_ropnet.size());
    } else if (forward) {
        out += MappedVector(m_ropf.data(), m_ropf.size());
    } else {
        out += MappedVector(m_ropr.data(), m_ropr.size());
    }

    thermo().setState_TP(T, P);
    updateROP();
    if (forward && reverse) {
        out -= MappedVector(m_ropnet.data(), m_ropnet.size());
    } else if (forward) {
        out -= MappedVector(m_ropf.data(), m_ropf.size());
    } else {
        out -= MappedVector(m_ropr.data(), m_ropr.size());
    }

    out *= dTinv;
    return out;
}

Eigen::VectorXd GasKinetics::ratesOfProgress_ddT_constV(bool forward, bool warn)
{
    double dTinv = 1. / m_jac_atol_deltaT;

    Eigen::VectorXd out(nReactions());
    vector_fp& k0 = m_rbuf0;
    vector_fp& k1 = m_rbuf1;

    if (warn && legacy_rate_constants_used()) {
        // @TODO  This is somewhat restrictive; however, using the alternative
        // definition by default appears to be inconsistent.
        warn_user("GasKinetics::ratesOfProgress_ddT_constV",
            "This routine relies on rate\nconstant calculations; here, the legacy "
            "definition introduces spurious temperature dependencies\ndue to the "
            "inclusion of third-body concentrations for ThreeBodyReaction objects.\n"
            "Proceed with caution, or set 'use_legacy_rate_constants' to false for "
            "new behavior.");
    }

    double T = thermo().temperature();
    double P = thermo().pressure();

    thermo().setState_TP(T + m_jac_atol_deltaT, P);
    updateROP();
    if (forward) {
        getFwdRateConstants(k1.data());
    } else {
        getRevRateConstants(k1.data());
    }

    out.fill(0.);
    thermo().setState_TP(T, P);
    updateROP();
    if (forward) {
        out += MappedVector(m_ropf.data(), m_ropf.size());
        getFwdRateConstants(k0.data());
    } else {
        out += MappedVector(m_ropr.data(), m_ropr.size());
        getRevRateConstants(k0.data());
    }

    for (size_t i = 0; i < k0.size(); i++) {
        if (k0[i] != 0) {
            out(i) *= dTinv * (k1[i] - k0[i]) / k0[i];
        } // else not needed: out(i) already zero
    }
    return out;
}

void GasKinetics::processConcentrations_ddTscaled(double* rop)
{
    double ddT_scaled;
    if (thermo().type() == "IdealGas") {
        ddT_scaled = -1. / thermo().temperature();
    } else {
        double T = thermo().temperature();
        double P = thermo().pressure();
        thermo().setState_TP(T + m_jac_atol_deltaT, P);
        double ctot1 = thermo().molarDensity();
        thermo().setState_TP(T, P);
        double ctot0 = thermo().molarDensity();
        ddT_scaled = (ctot1 / ctot0 - 1) / m_jac_atol_deltaT;
    }
    for (size_t i = 0; i < nReactions(); ++i ) {
        rop[i] *= ddT_scaled;
    }
}

Eigen::VectorXd GasKinetics::fwdRatesOfProgress_ddT()
{
    if (!m_jac_exact_ddT) {
        if (m_jac_const_pressure) {
            return ratesOfProgress_ddT_constP(true, false);
        }
        return ratesOfProgress_ddT_constV(true);
    }

    updateROP();
    Eigen::VectorXd dFwdRop(nReactions());
    copy(m_ropf.begin(), m_ropf.end(), &dFwdRop[0]);
    for (auto& rates : m_bulk_rates) {
        rates->processRateConstants_ddTscaled(
            thermo(), dFwdRop.data(), m_concm.data());
    }

    for (size_t i = 0; i < m_legacy.size(); ++i) {
        dFwdRop(m_legacy[i]) = NAN;
    }

    if (m_jac_const_pressure) {
        MappedVector dFwdRopC(m_rbuf1.data(), nReactions());
        dFwdRopC.fill(0.);
        m_reactantStoich.scale(m_ropf.data(), dFwdRopC.data());

        // multiply rop by enhanced 3b conc for all 3b rxns
        if (!concm_3b_values.empty()) {
            // Do not support legacy CTI/XML-based reaction rate evaluators
            throw CanteraError("GasKinetics::fwdRatesOfProgress_ddT",
                "Not supported for legacy input format.");
        }

        // reactions involving third body
        if (!concm_multi_values.empty()) {
            MappedVector dFwdRopM(m_rbuf2.data(), nReactions());
            dFwdRopM.fill(0.);
            m_multi_concm.scale(m_ropf.data(), dFwdRopM.data());
            dFwdRopC += dFwdRopM;
        }

        // add term to account for changes of concentrations
        processConcentrations_ddTscaled(dFwdRopC.data());
        dFwdRop += dFwdRopC;
    }

    return dFwdRop;
}

void GasKinetics::processEquilibriumConstants_ddTscaled(double* drkcn)
{
    double dTinv = 1. / m_jac_atol_deltaT;
    vector_fp& kc0 = m_rbuf0;
    vector_fp& kc1 = m_rbuf1;

    double T = thermo().temperature();
    double P = thermo().pressure();
    thermo().setState_TP(T + m_jac_atol_deltaT, P);
    getEquilibriumConstants(kc1.data());

    thermo().setState_TP(T, P);
    getEquilibriumConstants(kc0.data());

    for (size_t i = 0; i < nReactions(); ++i) {
        drkcn[i] *= (kc0[i] - kc1[i]) * dTinv;
        drkcn[i] /= kc0[i]; // divide once as this is a scaled derivative
    }

    for (size_t i = 0; i < m_irrev.size(); ++i) {
        drkcn[m_irrev[i]] = 0.0;
    }
}

Eigen::VectorXd GasKinetics::revRatesOfProgress_ddT()
{
    if (!m_jac_exact_ddT) {
        if (m_jac_const_pressure) {
            return ratesOfProgress_ddT_constP(false, true);
        }
        return ratesOfProgress_ddT_constV(false);
    }

    // reverse rop times scaled rate constant derivative
    updateROP();
    Eigen::VectorXd dRevRop(nReactions());
    copy(m_ropr.begin(), m_ropr.end(), &dRevRop[0]);
    for (auto& rates : m_bulk_rates) {
        rates->processRateConstants_ddTscaled(
            thermo(), dRevRop.data(), m_concm.data());
    }

    // reverse rop times scaled inverse equilibrium constant derivatives
    MappedVector dRevRop2(m_rbuf2.data(), nReactions());
    copy(m_ropr.begin(), m_ropr.end(), m_rbuf2.begin());
    processEquilibriumConstants_ddTscaled(dRevRop2.data());

    for (size_t i = 0; i < m_legacy.size(); ++i) {
        dRevRop(m_legacy[i]) = NAN;
    }

    dRevRop += dRevRop2;

    if (m_jac_const_pressure) {
        MappedVector dRevRopC(m_rbuf1.data(), nReactions());
        dRevRopC.fill(0.);
        m_revProductStoich.scale(m_ropr.data(), dRevRopC.data());

        // multiply rop by enhanced 3b conc for all 3b rxns
        if (!concm_3b_values.empty()) {
            // Do not support legacy CTI/XML-based reaction rate evaluators
            throw CanteraError("GasKinetics::revRatesOfProgress_ddT",
                "Not supported for legacy input format.");
        }

        // reactions involving third body
        if (!concm_multi_values.empty()) {
            MappedVector dRevRopM(m_rbuf2.data(), nReactions());
            dRevRopM.fill(0.);
            m_multi_concm.scale(m_ropr.data(), dRevRopM.data());
            dRevRopC += dRevRopM;
        }

        // add term to account for changes of concentrations
        processConcentrations_ddTscaled(dRevRopC.data());
        dRevRop += dRevRopC;
    }

    return dRevRop;
}

Eigen::VectorXd GasKinetics::netRatesOfProgress_ddT()
{
    if (!m_jac_exact_ddT) {
        if (m_jac_const_pressure) {
            return ratesOfProgress_ddT_constP(true, true);
        }
        return ratesOfProgress_ddT_constV(true)
            - ratesOfProgress_ddT_constV(false, false);
    }

    return fwdRatesOfProgress_ddT() - revRatesOfProgress_ddT();
}


bool GasKinetics::addReaction(shared_ptr<Reaction> r, bool finalize)
{
    // operations common to all reaction types
    bool added = BulkKinetics::addReaction(r, finalize);
    if (!added) {
        return false;
    } else if (!(r->usesLegacy())) {
        // Rate object already added in BulkKinetics::addReaction
        return true;
    }

    if (r->type() == "elementary-legacy") {
        addElementaryReaction(dynamic_cast<ElementaryReaction2&>(*r));
    } else if (r->type() == "three-body-legacy") {
        addThreeBodyReaction(dynamic_cast<ThreeBodyReaction2&>(*r));
    } else if (r->type() == "falloff") {
        addFalloffReaction(dynamic_cast<FalloffReaction&>(*r));
    } else if (r->type() == "chemically-activated") {
        addFalloffReaction(dynamic_cast<FalloffReaction&>(*r));
    } else if (r->type() == "pressure-dependent-Arrhenius-legacy") {
        addPlogReaction(dynamic_cast<PlogReaction2&>(*r));
    } else if (r->type() == "Chebyshev-legacy") {
        addChebyshevReaction(dynamic_cast<ChebyshevReaction2&>(*r));
    } else if (r->type() == "Blowers-Masel") {
        addBlowersMaselReaction(dynamic_cast<BlowersMaselReaction&>(*r));
    } else {
        throw CanteraError("GasKinetics::addReaction",
            "Unknown reaction type specified: '{}'", r->type());
    }
    m_legacy.push_back(nReactions() - 1);
    return true;
}

void GasKinetics::addFalloffReaction(FalloffReaction& r)
{
    // install high and low rate coeff calculators and extend the high and low
    // rate coeff value vectors
    size_t nfall = m_falloff_high_rates.nReactions();
    m_falloff_high_rates.install(nfall, r.high_rate);
    m_rfn_high.push_back(0.0);
    m_falloff_low_rates.install(nfall, r.low_rate);
    m_rfn_low.push_back(0.0);

    // add this reaction number to the list of falloff reactions
    m_fallindx.push_back(nReactions()-1);
    m_rfallindx[nReactions()-1] = nfall;

    // install the enhanced third-body concentration calculator
    map<size_t, double> efficiencies;
    for (const auto& eff : r.third_body.efficiencies) {
        size_t k = kineticsSpeciesIndex(eff.first);
        if (k != npos) {
            efficiencies[k] = eff.second;
        }
    }
    m_falloff_concm.install(nfall, efficiencies,
                            r.third_body.default_efficiency);
    concm_falloff_values.resize(m_falloff_concm.workSize());

    // install the falloff function calculator for this reaction
    m_falloffn.install(nfall, r.type(), r.falloff);
    falloff_work.resize(m_falloffn.workSize());
}

void GasKinetics::addThreeBodyReaction(ThreeBodyReaction2& r)
{
    m_rates.install(nReactions()-1, r.rate);
    map<size_t, double> efficiencies;
    for (const auto& eff : r.third_body.efficiencies) {
        size_t k = kineticsSpeciesIndex(eff.first);
        if (k != npos) {
            efficiencies[k] = eff.second;
        }
    }
    m_3b_concm.install(nReactions()-1, efficiencies,
                       r.third_body.default_efficiency);
    concm_3b_values.resize(m_3b_concm.workSize());
}

void GasKinetics::addPlogReaction(PlogReaction2& r)
{
    m_plog_rates.install(nReactions()-1, r.rate);
}

void GasKinetics::addChebyshevReaction(ChebyshevReaction2& r)
{
    m_cheb_rates.install(nReactions()-1, r.rate);
}

void GasKinetics::addBlowersMaselReaction(BlowersMaselReaction& r)
{
    m_blowersmasel_rates.install(nReactions()-1, r.rate);
}

void GasKinetics::modifyReaction(size_t i, shared_ptr<Reaction> rNew)
{
    // operations common to all bulk reaction types
    BulkKinetics::modifyReaction(i, rNew);

    if (!(rNew->usesLegacy())) {
        // Rate object already modified in BulkKinetics::modifyReaction
        return;
    }

    if (rNew->type() == "elementary-legacy") {
        modifyElementaryReaction(i, dynamic_cast<ElementaryReaction2&>(*rNew));
    } else if (rNew->type() == "three-body-legacy") {
        modifyThreeBodyReaction(i, dynamic_cast<ThreeBodyReaction2&>(*rNew));
    } else if (rNew->type() == "falloff") {
        modifyFalloffReaction(i, dynamic_cast<FalloffReaction&>(*rNew));
    } else if (rNew->type() == "chemically-activated") {
        modifyFalloffReaction(i, dynamic_cast<FalloffReaction&>(*rNew));
    } else if (rNew->type() == "pressure-dependent-Arrhenius-legacy") {
        modifyPlogReaction(i, dynamic_cast<PlogReaction2&>(*rNew));
    } else if (rNew->type() == "Chebyshev-legacy") {
        modifyChebyshevReaction(i, dynamic_cast<ChebyshevReaction2&>(*rNew));
    } else if (rNew->type() == "Blowers-Masel") {
        modifyBlowersMaselReaction(i, dynamic_cast<BlowersMaselReaction&>(*rNew));
    } else {
        throw CanteraError("GasKinetics::modifyReaction",
            "Unknown reaction type specified: '{}'", rNew->type());
    }

    // invalidate all cached data
    m_ROP_ok = false;
    m_temp += 0.1234;
    m_pres += 0.1234;
}

void GasKinetics::modifyThreeBodyReaction(size_t i, ThreeBodyReaction2& r)
{
    m_rates.replace(i, r.rate);
}

void GasKinetics::modifyFalloffReaction(size_t i, FalloffReaction& r)
{
    size_t iFall = m_rfallindx[i];
    m_falloff_high_rates.replace(iFall, r.high_rate);
    m_falloff_low_rates.replace(iFall, r.low_rate);
    m_falloffn.replace(iFall, r.falloff);
}

void GasKinetics::modifyPlogReaction(size_t i, PlogReaction2& r)
{
    m_plog_rates.replace(i, r.rate);
}

void GasKinetics::modifyChebyshevReaction(size_t i, ChebyshevReaction2& r)
{
    m_cheb_rates.replace(i, r.rate);
}

void GasKinetics::modifyBlowersMaselReaction(size_t i, BlowersMaselReaction& r)
{
    m_blowersmasel_rates.replace(i, r.rate);
}

void GasKinetics::init()
{
    BulkKinetics::init();
    m_logp_ref = log(thermo().refPressure()) - log(GasConstant);
}

void GasKinetics::invalidateCache()
{
    BulkKinetics::invalidateCache();
    m_pres += 0.13579;
}

}
