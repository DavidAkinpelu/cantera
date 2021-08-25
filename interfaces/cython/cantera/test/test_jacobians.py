import numpy as np

import cantera as ct
from . import utilities


class RateExpressionTests:
    # Generic test class to check Jacobians evaluated for a single reaction within
    # a reaction mechanism

    rxn_idx = None # index of reaction to be tested
    phase = None
    rtol = 1e-5
    orders = None
    ix3b = [] # three-body indices

    @classmethod
    def setUpClass(cls):
        ct.use_legacy_rate_constants(False)
        cls.tpx = cls.gas.TPX

        cls.r_stoich = cls.gas.reactant_stoich_coefficients
        cls.p_stoich = cls.gas.product_stoich_coefficients

        cls.rxn = cls.gas.reactions()[cls.rxn_idx]
        cls.rix = [cls.gas.species_index(k) for k in cls.rxn.reactants.keys()]
        cls.pix = [cls.gas.species_index(k) for k in cls.rxn.products.keys()]

    def setUp(self):
        self.gas.TPX = self.tpx
        self.gas.set_multiplier(0.)
        self.gas.set_multiplier(1., self.rxn_idx)

        self.gas.jacobian_settings = {} # reset defaults
        self.gas.jacobian_settings = {"mole-fractions": False} # use concentrations

    def rop_ddC(self, spc_ix, mode, rtol_deltac=1e-6, atol_deltac=1e-20, fraction=False):
        # numerical derivative for rates-of-progress with respect to mole fractions
        def calc(fwd):
            if mode == "forward":
                return self.gas.forward_rates_of_progress
            if mode == "reverse":
                return self.gas.reverse_rates_of_progress
            if mode == "net":
                return self.gas.net_rates_of_progress

        rop = calc(mode)
        kf = self.gas.forward_rate_constants[self.rxn_idx]
        n_spc = self.gas.n_species
        dconc = np.zeros((n_spc))
        dconc[spc_ix] = self.gas.concentrations[spc_ix] * rtol_deltac + atol_deltac
        xnew = (self.gas.concentrations + dconc) / self.gas.density_mole
        # adjust pressure to compensate for concentration change
        pnew = self.tpx[1] + dconc[spc_ix] * ct.gas_constant * self.gas.T
        self.gas.TPX = self.tpx[0], pnew, xnew
        kfnew = self.gas.forward_rate_constants[self.rxn_idx] # compensate for pressure adjustment
        drop = (calc(mode) * kf / kfnew - rop) / dconc[spc_ix]
        self.gas.TPX = self.tpx

        if fraction:
            drop *= self.gas.density_mole
        return drop

    def rop_ddT(self, mode=None, const_p=False, dt=1e-6):
        # numerical derivative for rates-of-progress at constant pressure
        def calc(fwd):
            if mode == "forward":
                return self.gas.forward_rates_of_progress, self.gas.forward_rate_constants
            if mode == "reverse":
                return self.gas.reverse_rates_of_progress, self.gas.reverse_rate_constants
            if mode == "net":
                return self.gas.net_rates_of_progress, None
            return None, None

        self.gas.TP = self.tpx[0] + dt, self.tpx[1]
        rop1, k1 = calc(mode)
        self.gas.TP = self.tpx[:2]
        rop0, k0 = calc(mode)
        if const_p:
            return (rop1[self.rxn_idx] - rop0[self.rxn_idx]) / dt
        if k0[self.rxn_idx] == 0:
            return 0
        drop = rop0[self.rxn_idx] / k0[self.rxn_idx]
        drop *= (k1[self.rxn_idx] - k0[self.rxn_idx]) / dt
        return drop

    def rate_ddC(self, spc_ix, mode=None, rtol_deltac=1e-6, fraction=False):
        # numerical derivative for production rates with respect to mole fractions
        def calc(mode):
            if mode == "creation":
                return self.gas.creation_rates
            if mode == "destruction":
                return self.gas.destruction_rates
            if mode == "net":
                return self.gas.net_production_rates

        rate = calc(mode)
        kf = self.gas.forward_rate_constants[self.rxn_idx]
        n_spc = self.gas.n_species
        dconc = np.zeros((n_spc))
        dconc[spc_ix] = self.gas.concentrations[spc_ix] * rtol_deltac
        xnew = (self.gas.concentrations + dconc) / self.gas.density_mole
        # adjust pressure to compensate for concentration change
        pnew = self.tpx[1] + dconc[spc_ix] * ct.gas_constant * self.gas.T
        self.gas.TPX = self.tpx[0], pnew, xnew
        kfnew = self.gas.forward_rate_constants[self.rxn_idx] # compensate for pressure adjustment
        drate = (calc(mode) * kf / kfnew - rate) / dconc[spc_ix]
        self.gas.TPX = self.tpx

        if fraction:
            drate *= self.gas.density_mole
        return drate

    def test_stoich_coeffs(self):
        # check stoichiometric coefficient output
        for k, v in self.rxn.reactants.items():
            ix = self.gas.species_index(k)
            self.assertEqual(self.r_stoich[ix, self.rxn_idx], v)
        for k, v in self.rxn.products.items():
            ix = self.gas.species_index(k)
            self.assertEqual(self.p_stoich[ix, self.rxn_idx], v)

    def test_forward_rop_ddC_basic(self):
        # ensure that non-zero entries are where they are expected
        drop = self.gas.forward_rop_species_derivatives
        for spc_ix in set(self.rix + self.ix3b):
            self.assertTrue(drop[self.rxn_idx, spc_ix]) # non-zero
            drop[self.rxn_idx, spc_ix] = 0
        if drop.any():
            print(drop[:self.rxn_idx,:])
            print(drop[self.rxn_idx,:])
            print(drop[self.rxn_idx + 1:,:])
        self.assertFalse(drop.any())

        drop = self.gas.forward_rop_temperature_derivatives
        self.assertTrue(drop[self.rxn_idx]) # non-zero
        drop[self.rxn_idx] = 0
        self.assertFalse(drop.any())

    def test_forward_rop_ddC(self):
        # check derivatives of forward rates of progress with respect to mole fractions
        # against analytic result
        self.gas.jacobian_settings = {"skip-third-bodies": True}
        drop = self.gas.forward_rop_species_derivatives
        rop = self.gas.forward_rates_of_progress
        for spc_ix in self.rix:
            if self.orders is None:
                stoich = self.r_stoich[spc_ix, self.rxn_idx]
            else:
                stoich = self.orders[self.gas.species_names[spc_ix]]
            self.assertNear(rop[self.rxn_idx],
                drop[self.rxn_idx, spc_ix] * self.gas.concentrations[spc_ix] / stoich)

    def test_forward_rop_ddC_num(self):
        # check derivatives of forward rates of progress with respect to mole fractions
        # against numeric result
        drop = self.gas.forward_rop_species_derivatives
        for spc_ix in self.rix:
            drop_num = self.rop_ddC(spc_ix, mode="forward")
            self.assertArrayNear(drop[:, spc_ix], drop_num, self.rtol)

    def test_forward_rop_ddX_num(self):
        # check derivatives of forward rates of progress with respect to mole fractions
        # against numeric result
        self.gas.jacobian_settings = {"mole-fractions": True}
        drop = self.gas.forward_rop_species_derivatives
        for spc_ix in self.rix:
            drop_num = self.rop_ddC(spc_ix, mode="forward", fraction=True)
            self.assertArrayNear(drop[:, spc_ix], drop_num, self.rtol)

    def test_reverse_rop_basic(self):
        # ensure that non-zero entries are where they are expected
        if not self.rxn.reversible:
            return
        drop = self.gas.reverse_rop_species_derivatives
        for spc_ix in set(self.pix + self.ix3b):
            self.assertTrue(drop[self.rxn_idx, spc_ix]) # non-zero
            drop[self.rxn_idx, spc_ix] = 0
        self.assertFalse(drop.any())

        drop = self.gas.reverse_rop_temperature_derivatives
        self.assertTrue(drop[self.rxn_idx]) # non-zero
        drop[self.rxn_idx] = 0
        self.assertFalse(drop.any())

    def test_reverse_rop_ddC(self):
        # check derivatives of reverse rates of progress with respect to mole fractions
        # against analytic result
        self.gas.jacobian_settings = {"skip-third-bodies": True}
        drop = self.gas.reverse_rop_species_derivatives
        rop = self.gas.reverse_rates_of_progress
        for spc_ix in self.pix:
            stoich = self.p_stoich[spc_ix, self.rxn_idx]
            self.assertNear(rop[self.rxn_idx],
                drop[self.rxn_idx, spc_ix] * self.gas.concentrations[spc_ix] / stoich)

    def test_reverse_rop_ddC_num(self):
        # check derivatives of reverse rates of progress with respect to mole fractions
        # against numeric result
        drop = self.gas.reverse_rop_species_derivatives
        for spc_ix in self.pix:
            drop_num = self.rop_ddC(spc_ix, mode="reverse")
            self.assertArrayNear(drop[:, spc_ix], drop_num, self.rtol)

    def test_reverse_rop_ddX_num(self):
        # check derivatives of reverse rates of progress with respect to mole fractions
        # against numeric result
        self.gas.jacobian_settings = {"mole-fractions": True}
        drop = self.gas.reverse_rop_species_derivatives
        for spc_ix in self.pix:
            drop_num = self.rop_ddC(spc_ix, mode="reverse", fraction=True)
            self.assertArrayNear(drop[:, spc_ix], drop_num, self.rtol)

    def test_net_rop(self):
        # ensure that non-zero entries are where they are expected
        if not self.rxn.reversible:
            return
        drop = self.gas.net_rop_species_derivatives
        for spc_ix in set(self.rix + self.pix + self.ix3b):
            self.assertTrue(drop[self.rxn_idx, spc_ix]) # non-zero
            drop[self.rxn_idx, spc_ix] = 0
        self.assertFalse(drop.any())

        drop = self.gas.net_rop_temperature_derivatives
        self.assertTrue(drop[self.rxn_idx]) # non-zero
        drop[self.rxn_idx] = 0
        self.assertFalse(drop.any())

    def test_net_rop_ddC_num(self):
        # check derivatives of net rates of progress with respect to mole fractions
        # against numeric result
        drop = self.gas.net_rop_species_derivatives
        for spc_ix in self.rix + self.pix:
            drop_num = self.rop_ddC(spc_ix, mode="net")
            ix = drop[:, spc_ix] != 0
            self.assertArrayNear(drop[ix, spc_ix], drop_num[ix], self.rtol)

    def test_creation_ddC_num(self):
        # check derivatives of creation rates with respect to mole fractions
        drate = self.gas.creation_rate_species_derivatives
        for spc_ix in self.rix + self.pix:
            drate_num = self.rate_ddC(spc_ix, "creation")
            ix = drate[:, spc_ix] != 0
            self.assertArrayNear(drate[ix, spc_ix], drate_num[ix], self.rtol)

    def test_destruction_ddC_num(self):
        # check derivatives of destruction rates with respect to mole fractions
        drate = self.gas.destruction_rate_species_derivatives
        for spc_ix in self.rix + self.pix:
            drate_num = self.rate_ddC(spc_ix, "destruction")
            ix = drate[:, spc_ix] != 0
            self.assertArrayNear(drate[ix, spc_ix], drate_num[ix], self.rtol)

    def test_net_production_ddC_num(self):
        # check derivatives of destruction rates with respect to mole fractions
        drate = self.gas.net_production_rate_species_derivatives
        for spc_ix in self.rix + self.pix:
            drate_num = self.rate_ddC(spc_ix, "net")
            ix = drate[:, spc_ix] != 0
            self.assertArrayNear(drate[ix, spc_ix], drate_num[ix], self.rtol)

    def test_exact_rate_ddT(self):
        # check basic temperature derivative of rate coefficient
        rate = self.rxn.rate
        if rate.__class__.__name__ != "ArrheniusRate":
            return
        T = self.tpx[0]

        R = ct.gas_constant
        Ea = rate.activation_energy
        b =  rate.temperature_exponent
        A = rate.pre_exponential_factor
        k0 = rate(T)
        self.assertNear(k0, A * T**b * np.exp(-Ea/R/T))

        scaled_ddT = (Ea / R / T + b) / T
        dkdT = rate.ddT(T)
        self.assertNear(dkdT, k0 * scaled_ddT) # exact

        dT = 1e-6
        k1 = rate(T + dT)
        self.assertNear((k1 - k0) / dT, dkdT, 1e-6) # numeric

        rop = self.gas.forward_rates_of_progress
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": True,
            "constant-pressure": False,
        }
        drop = self.gas.forward_rop_temperature_derivatives
        self.assertNear(drop[self.rxn_idx], rop[self.rxn_idx] * scaled_ddT) # exact

    def test_forward_rop_constV_ddT(self):
        # compare exact and approximate forward rate of progress derivatives
        self.gas.jacobian_settings = {
            "constant-pressure": False,
        }
        drop_approx = self.gas.forward_rop_temperature_derivatives
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": True,
            "constant-pressure": False,
        }
        drop = self.gas.forward_rop_temperature_derivatives
        self.assertNear(drop[self.rxn_idx], drop_approx[self.rxn_idx], self.rtol)

    def test_forward_rop_ddT_constV_num(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "constant-pressure": False,
        }
        drop_approx = self.gas.forward_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="forward")
        self.assertNear(drop_approx[self.rxn_idx], drop_num, self.rtol)

    def test_forward_rop_ddT_constP(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": True,
            "constant-pressure": True,
        }
        drop = self.gas.forward_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="forward", const_p=True)
        self.assertNear(drop[self.rxn_idx], drop_num, self.rtol)

    def test_forward_rop_ddT_constP_num(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": False,
            "constant-pressure": True,
        }
        drop = self.gas.forward_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="forward", const_p=True)
        self.assertNear(drop[self.rxn_idx], drop_num, self.rtol)

    def test_reverse_rop_constV_ddT(self):
        # compare exact and approximate reverse rate of progress derivatives
        self.gas.jacobian_settings = {
            "constant-pressure": False,
        }
        drop_approx = self.gas.reverse_rop_temperature_derivatives
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": True,
            "constant-pressure": False,
        }
        drop = self.gas.reverse_rop_temperature_derivatives
        self.assertNear(drop[self.rxn_idx], drop_approx[self.rxn_idx], self.rtol)

    def test_reverse_rop_ddT_constV_num(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "constant-pressure": False,
        }
        drop_approx = self.gas.reverse_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="reverse")
        self.assertNear(drop_approx[self.rxn_idx], drop_num, self.rtol)

    def test_reverse_rop_constP_ddT(self):
        # compare exact and approximate reverse rate of progress derivatives
        drop_approx = self.gas.reverse_rop_temperature_derivatives
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": True,
        }
        drop = self.gas.reverse_rop_temperature_derivatives
        self.assertNear(drop[self.rxn_idx], drop_approx[self.rxn_idx], self.rtol)

    def test_reverse_rop_ddT_constP_num(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "constant-pressure": True,
        }
        drop_approx = self.gas.reverse_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="reverse", const_p=True)
        self.assertNear(drop_approx[self.rxn_idx], drop_num, self.rtol)

    def test_net_rop_constV_ddT(self):
        # compare exact and approximate reverse rate of progress derivatives
        self.gas.jacobian_settings = {
            "constant-pressure": False,
        }
        drop_approx = self.gas.net_rop_temperature_derivatives
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": True,
            "constant-pressure": False,
        }
        drop = self.gas.net_rop_temperature_derivatives
        self.assertNear(drop[self.rxn_idx], drop_approx[self.rxn_idx], self.rtol)

    def test_net_rop_ddT_constV_num(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "constant-pressure": False,
        }
        drop_approx = self.gas.net_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="forward") - self.rop_ddT(mode="reverse")
        self.assertNear(drop_approx[self.rxn_idx], drop_num, self.rtol)

    def test_net_rop_constP_ddT(self):
        # compare exact and approximate reverse rate of progress derivatives
        self.gas.jacobian_settings = {
            "constant-pressure": True,
            "exact-temperature-derivatives": False,
        }
        drop_approx = self.gas.net_rop_temperature_derivatives
        self.gas.jacobian_settings = {
            "constant-pressure": True,
            "exact-temperature-derivatives": True,
        }
        drop = self.gas.net_rop_temperature_derivatives
        self.assertNear(drop[self.rxn_idx], drop_approx[self.rxn_idx], self.rtol)

    def test_net_rop_ddT_constP_num(self):
        # check derivatives of foward rop with respect to temperature
        self.gas.jacobian_settings = {
            "exact-temperature-derivatives": False,
            "constant-pressure": True,
        }
        drop_approx = self.gas.net_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="net", const_p=True)
        # drop_num = self.rop_ddT(mode="forward", const_p=True) - self.rop_ddT(mode="reverse", const_p=True)
        self.assertNear(drop_approx[self.rxn_idx], drop_num, self.rtol)


class HydrogenOxygen(RateExpressionTests):

    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution("h2o2.yaml", transport_model=None)
        #   species: [H2, H, O, O2, OH, H2O, HO2, H2O2, AR, N2]
        cls.gas.X = [0.1, 1e-4, 1e-5, 0.2, 2e-4, 0.3, 1e-6, 5e-5, 0.3, 0.1]
        cls.gas.TP = 800, 2*ct.one_atm
        super().setUpClass()


class TestElementaryRev(HydrogenOxygen, utilities.CanteraTest):
    # Standard elementary reaction with two reactants
    # - equation: O + H2 <=> H + OH  # Reaction 3
    #   rate-constant: {A: 3.87e+04, b: 2.7, Ea: 6260.0}
    rxn_idx = 2


class TestElementarySelf(HydrogenOxygen, utilities.CanteraTest):
    # Elementary reaction with reactant reacting with itself
    # - equation: 2 HO2 <=> O2 + H2O2  # Reaction 28
    rxn_idx = 27


class TestFalloff(HydrogenOxygen, utilities.CanteraTest):
    # Fall-off reaction
    # - equation: 2 OH (+M) <=> H2O2 (+M)  # Reaction 22
    #   type: falloff
    #   low-P-rate-constant: {A: 2.3e+18, b: -0.9, Ea: -1700.0}
    #   high-P-rate-constant: {A: 7.4e+13, b: -0.37, Ea: 0.0}
    #   Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
    #   efficiencies: {H2: 2.0, H2O: 6.0, AR: 0.7}
    rxn_idx = 21

    def test_exact_rate_ddT(self):
        # Not implemented, as Falloff reactions still use legacy code
        pass


class TestThreeBody(HydrogenOxygen, utilities.CanteraTest):
    # Three body reaction with default efficiency
    # - equation: O + H + M <=> OH + M  # Reaction 2
    #   type: three-body
    #   rate-constant: {A: 5.0e+17, b: -1.0, Ea: 0.0}
    #   efficiencies: {H2: 2.0, H2O: 6.0, AR: 0.7}
    rxn_idx = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ix3b = list(range(cls.gas.n_species))

    def test_thirdbodies_forward(self):
        drop = self.gas.forward_rop_species_derivatives
        self.gas.jacobian_settings = {"skip-third-bodies": True}
        drops = self.gas.forward_rop_species_derivatives
        dropm = drop - drops
        rop = self.gas.forward_rates_of_progress
        self.assertNear(rop[self.rxn_idx],
            (dropm[self.rxn_idx] * self.gas.concentrations).sum())

    def test_thirdbodies_reverse(self):
        drop = self.gas.reverse_rop_species_derivatives
        self.gas.jacobian_settings = {"skip-third-bodies": True}
        drops = self.gas.reverse_rop_species_derivatives
        dropm = drop - drops
        rop = self.gas.reverse_rates_of_progress
        self.assertNear(rop[self.rxn_idx],
            (dropm[self.rxn_idx] * self.gas.concentrations).sum())


class EdgeCases(RateExpressionTests):

    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution("jacobian-tests.yaml", transport_model=None)
        #   species: [H2, H, O, O2, OH, H2O, HO2, H2O2, AR]
        cls.gas.X = [0.1, 1e-4, 1e-5, 0.2, 2e-4, 0.3, 1e-6, 5e-5, 0.4]
        cls.gas.TP = 800, 2*ct.one_atm
        super().setUpClass()


class TestElementaryIrr(EdgeCases, utilities.CanteraTest):
    # Irreversible elementary reaction with two reactants
    # - equation: O + HO2 => OH + O2  # Reaction 1
    #   rate-constant: {A: 2.0e+13, b: 0.0, Ea: 0.0}
    rxn_idx = 0


class TestElementaryOne(EdgeCases, utilities.CanteraTest):
    # Three-body reaction with single reactant species
    # - equation: H2 <=> H + H  # Reaction 2
    #   rate-constant: {A: 1.968e+16, b: 0.0, Ea: 9.252008e+04}
    rxn_idx = 1


class TestElementaryThree(EdgeCases, utilities.CanteraTest):
    # Elementary reaction with three reactants
    # - equation: OH + O + O2 <=> HO2 + O2  # Reaction 3
    #   rate-constant: {A: 2.08e+19, b: -1.24, Ea: 0.0}
    rxn_idx = 2


class TestElementaryFrac(EdgeCases, utilities.CanteraTest):
    # Elementary reaction with specified reaction order
    # - equation: 0.7 H2 + 0.6 OH + 0.2 O2 => H2O  # Reaction 4
    #   rate-constant: {A: 3.981072e+04, b: 0.0, Ea: 0.0 cal/mol}
    #   orders: {H2: 0.8, O2: 1.0, OH: 2.0}
    rxn_idx = 3
    orders = {"H2": 0.8, "O2": 1.0, "OH": 2.0}


class TestThreeBodyNoDefault(EdgeCases, utilities.CanteraTest):
    # Three body reaction without default efficiency
    # - equation: O + H + M <=> OH + M  # Reaction 2
    #   type: three-body
    #   rate-constant: {A: 5.0e+17, b: -1.0, Ea: 0.0}
    #   default-efficiency: 0.
    #   efficiencies: {H2: 2.0, H2O: 6.0, AR: 0.7}
    rxn_idx = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        efficiencies = {"H2": 2.0, "H2O": 6.0, "AR": 0.7}
        cls.ix3b = [cls.gas.species_index(k) for k in efficiencies.keys()]


class FromScratchCases(RateExpressionTests):

    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution("kineticsfromscratch.yaml", transport_model=None)
        #   species: [AR, O, H2, H, OH, O2, H2O, H2O2, HO2]
        cls.gas.X = [0.4, 1e-5, 0.1, 1e-4, 2e-4, 0.2, 0.3, 53-5, 1e-6]
        cls.gas.TP = 800, 2*ct.one_atm
        super().setUpClass()


class TestPlog(FromScratchCases, utilities.CanteraTest):
    # Plog reaction
    # - equation: H2 + O2 <=> 2 OH  # Reaction 4
    #   type: pressure-dependent-Arrhenius
    #   rate-constants:
    #   - {P: 0.01 atm, A: 1.2124e+16, b: -0.5779, Ea: 1.08727e+04}
    #   - {P: 1.0 atm, A: 4.9108e+31, b: -4.8507, Ea: 2.47728e+04}
    #   - {P: 10.0 atm, A: 1.2866e+47, b: -9.0246, Ea: 3.97965e+04}
    #   - {P: 100.0 atm, A: 5.9632e+56, b: -11.529, Ea: 5.25996e+04}
    rxn_idx = 3


class TestChebyshev(FromScratchCases, utilities.CanteraTest):
    # Chebyshev reaction
    # - equation: HO2 <=> OH + O  # Reaction 5
    #   type: Chebyshev
    #   temperature-range: [290.0, 3000.0]
    #   pressure-range: [9.869232667160128e-03 atm, 98.69232667160128 atm]
    #   data:
    #   - [8.2883, -1.1397, -0.12059, 0.016034]
    #   - [1.9764, 1.0037, 7.2865e-03, -0.030432]
    #   - [0.3177, 0.26889, 0.094806, -7.6385e-03]
    rxn_idx = 4
    rtol = 5e-4


class FullTests:
    # Generic test class to check Jacobians evaluated for an entire reaction mechanims
    rtol = 5e-5

    @classmethod
    def setUpClass(cls):
        ct.use_legacy_rate_constants(False)
        cls.tpx = cls.gas.TPX
        cls.gas.jacobian_settings = {
            "exact-temperature-derivatives": False,
            "constant-pressure": True,
            "mole-fractions": True,
        }

    def setUp(self):
        self.gas.TPX = self.tpx

    def rop_ddC(self, mode, rtol_deltac=1e-9, atol_deltac=1e-20):
        # numerical derivative for rates-of-progress with respect to mole fractions
        def calc(fwd):
            if mode == "forward":
                return self.gas.forward_rates_of_progress
            if mode == "reverse":
                return self.gas.reverse_rates_of_progress
            if mode == "net":
                return self.gas.net_rates_of_progress

        rop = calc(mode)
        n_spc, n_rxn = self.gas.n_species, self.gas.n_reactions
        dconc = np.zeros((n_spc))
        drop = np.zeros((n_rxn, n_spc))
        kf = self.gas.forward_rate_constants
        for spc_ix in range(n_spc):
            dconc *= 0
            dconc[spc_ix] = self.gas.concentrations[spc_ix] * rtol_deltac + atol_deltac
            xnew = (self.gas.concentrations + dconc) / self.gas.density_mole
            # adjust pressure to compensate for concentration change
            pnew = self.tpx[1] + dconc[spc_ix] * ct.gas_constant * self.gas.T
            self.gas.TPX = self.tpx[0], pnew, xnew
            kfnew = self.gas.forward_rate_constants # compensate for pressure adjustment
            kfnew[kfnew == 0] = 1.
            drop[:, spc_ix] = (calc(mode) * kf / kfnew - rop) / dconc[spc_ix]
            self.gas.TPX = self.tpx

        return drop * self.gas.density_mole

    def rop_ddT(self, mode=None, dt=1e-6):
        # numerical derivative for rates-of-progress at constant pressure
        def calc(fwd):
            if mode == "forward":
                return self.gas.forward_rates_of_progress
            if mode == "reverse":
                return self.gas.reverse_rates_of_progress
            if mode == "net":
                return self.gas.net_rates_of_progress
            return None

        self.gas.TP = self.tpx[0] + dt, self.tpx[1]
        rop1 = calc(mode)
        self.gas.TP = self.tpx[:2]
        rop0 = calc(mode)
        return (rop1 - rop0) / dt

    def test_forward_rop_ddC(self):
        # check forward rop against numerical jacobian with respect to concentrations
        drop = self.gas.forward_rop_species_derivatives
        drop_num = self.rop_ddC(mode="forward")
        stoich = self.gas.reactant_stoich_coefficients
        for i in range(self.gas.n_reactions):
            try:
                ix = stoich[:, i] != 0
                self.assertArrayNear(drop[i, ix], drop_num[i, ix], self.rtol)
            except AssertionError as err:
                print(self.gas.reaction(i))
                print(drop[i])
                print(drop_num[i])
                raise err

    def test_forward_rop_ddT(self):
        # check forward rop against numerical jacobian with respect to temperature
        drop = self.gas.forward_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="forward")
        self.assertArrayNear(drop, drop_num, self.rtol)

    def test_reverse_rop_ddC(self):
        # check reverse rop against numerical jacobian with respect to concentrations
        drop = self.gas.reverse_rop_species_derivatives
        drop_num = self.rop_ddC(mode="reverse")
        stoich = self.gas.product_stoich_coefficients
        for i in range(self.gas.n_reactions):
            try:
                ix = stoich[:, i] != 0
                self.assertArrayNear(drop[i, ix], drop_num[i, ix], self.rtol)
            except AssertionError as err:
                print(self.gas.reaction(i))
                print(drop[i])
                print(drop_num[i])
                raise err

    def test_reverse_rop_ddT(self):
        # check reverse rop against numerical jacobian with respect to temperature
        drop = self.gas.reverse_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="reverse")
        self.assertArrayNear(drop, drop_num, self.rtol)

    def test_net_rop_ddC(self):
        # check net rop against numerical jacobian with respect to concentrations
        drop = self.gas.net_rop_species_derivatives
        drop_num = self.rop_ddC(mode="net")
        stoich = self.gas.product_stoich_coefficients - self.gas.reactant_stoich_coefficients
        for i in range(self.gas.n_reactions):
            try:
                ix = stoich[:, i] != 0
                self.assertArrayNear(drop[i, ix], drop_num[i, ix], self.rtol)
            except AssertionError as err:
                if self.gas.reaction(i).reversible:
                    print(self.gas.reaction(i))
                    print(drop[i])
                    print(drop_num[i])
                    raise err

    def test_net_rop_ddT(self):
        # check net rop against numerical jacobian with respect to temperature
        drop = self.gas.net_rop_temperature_derivatives
        drop_num = self.rop_ddT(mode="net")
        self.assertArrayNear(drop, drop_num, self.rtol)


class FullHydrogenOxygen(FullTests, utilities.CanteraTest):

    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution("h2o2.yaml", transport_model=None)
        cls.gas.TPX = 300, 5 * ct.one_atm, "H2:1, O2:3"
        cls.gas.equilibrate("HP")
        super().setUpClass()


class FullGriMech(FullTests, utilities.CanteraTest):

    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution("gri30.yaml", transport_model=None)
        cls.gas.TPX = 300, ct.one_atm, "CH4:1, O2:3"
        cls.gas.equilibrate("HP")
        super().setUpClass()


class FullEdgeCases(FullTests, utilities.CanteraTest):

    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution("jacobian-tests.yaml", transport_model=None)
        #   species: [H2, H, O, O2, OH, H2O, HO2, H2O2, AR]
        cls.gas.X = [0.1, 1e-4, 1e-5, 0.2, 2e-4, 0.3, 1e-6, 5e-5, 0.4]
        cls.gas.TP = 800, 2*ct.one_atm
        super().setUpClass()
