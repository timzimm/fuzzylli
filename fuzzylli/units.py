import numpy as np

import fuzzylli.cosmology as cosmo


class Units:
    """
    A very simple unit conversion helper class providing conversion factors
    from/to code units to common astrophysical units beyond cgs.
    """

    cm = 1
    g = 1
    s = 1
    pc = 3.0856775815e18  # parsec in cm
    Msun = 1.988409871e33  # solar mass in gram
    m22 = 1.78266192e-55  # axion mass scale in gram
    yr = 60 * 60 * 24 * 365.25  # year in seconds
    kms = 1.0e5  # velocity in km/s
    Kpc = 1.0e3 * pc  # kiloparsec
    Mpc = 1.0e6 * pc  # megaparsec
    c_light = 2.99792458e10  # speed of light
    Myr = 1.0e6 * yr  # megayear
    Gyr = 1.0e9 * yr  # megayear
    Kpc_kms = Kpc * kms  # angular momentum
    Msun_per_pc = Msun / pc  # line density
    Msun_per_Kpc = Msun / Kpc  # line density
    Msun_per_Mpc = Msun / Mpc  # line density
    Msun_per_pc3 = Msun / (pc * pc * pc)  # volume density
    Msun_per_Kpc3 = Msun / (Kpc * Kpc * Kpc)  # volume density
    Msun_per_Mpc3 = Msun / (Mpc * Mpc * Mpc)  # volume density
    Gev_per_cm3 = 1.782662e-24  # volume density in Gev/cm^3
    hbar = 1.0545919e-27  # planks constant
    eV = 1.60218e-12  # erg
    Grav = 6.6743e-8

    def __init__(self, length_unit_in_cm, mass_unit_in_g, time_unit_in_s):
        self.__length_unit = length_unit_in_cm
        self.__mass_unit = mass_unit_in_g
        self.__time_unit = time_unit_in_s
        self.__action_unit = (
            self.__mass_unit * self.__length_unit**2 / self.__time_unit
        )
        self.from_cm = Units.cm / self.__length_unit
        self.from_g = Units.g / self.__mass_unit
        self.from_s = Units.s / self.__time_unit
        self.from_pc = Units.pc / self.__length_unit
        self.from_Msun = Units.Msun / self.__mass_unit
        self.from_Kpc = Units.Kpc / self.__length_unit
        self.from_Mpc = Units.Mpc / self.__length_unit
        self.from_yr = Units.yr / self.__time_unit
        self.from_Myr = Units.Myr / self.__time_unit
        self.from_Gyr = Units.Gyr / self.__time_unit
        self.from_kms = Units.kms / (self.__length_unit / self.__time_unit)
        self.from_Kpc_kms = Units.Kpc_kms / (
            self.__length_unit * self.__length_unit / self.__time_unit
        )
        self.from_Msun_per_pc = Units.Msun_per_pc / (
            self.__mass_unit / self.__length_unit
        )
        self.from_Msun_per_Kpc = Units.Msun_per_Kpc / (
            self.__mass_unit / self.__length_unit
        )
        self.from_Msun_per_Mpc = Units.Msun_per_Mpc / (
            self.__mass_unit / self.__length_unit
        )
        self.from_Msun_per_pc3 = Units.Msun_per_pc3 / (
            self.__mass_unit / self.__length_unit**3
        )
        self.from_Msun_per_Kpc3 = Units.Msun_per_Kpc3 / (
            self.__mass_unit / self.__length_unit**3
        )
        self.from_Msun_per_Mpc3 = Units.Msun_per_Mpc3 / (
            self.__mass_unit / self.__length_unit**3
        )
        self.from_Gev_per_cm3 = Units.Gev_per_cm3 / (
            self.__mass_unit / self.__length_unit**3
        )
        self.from_hbar = Units.hbar / self.__action_unit
        self.from_m22 = Units.m22 / self.__mass_unit

        self.to_cm = 1.0 / self.from_cm
        self.to_g = 1.0 / self.from_g
        self.to_s = 1.0 / self.from_s

        self.to_Msun = 1.0 / self.from_Msun
        self.to_m22 = 1.0 / self.from_m22
        self.to_pc = 1.0 / self.from_pc
        self.to_Kpc = 1.0 / self.from_Kpc
        self.to_Mpc = 1.0 / self.from_Mpc
        self.to_yr = 1.0 / self.from_yr
        self.to_Myr = 1.0 / self.from_Myr
        self.to_Gyr = 1.0 / self.from_Gyr
        self.to_kms = 1.0 / self.from_kms
        self.to_Kpc_kms = 1.0 / self.from_Kpc_kms
        self.to_Msun_per_pc = 1.0 / self.from_Msun_per_pc
        self.to_Msun_per_Kpc = 1.0 / self.from_Msun_per_Kpc
        self.to_Msun_per_Mpc = 1.0 / self.from_Msun_per_Mpc
        self.to_Msun_per_pc3 = 1.0 / self.from_Msun_per_pc3
        self.to_Msun_per_Kpc3 = 1.0 / self.from_Msun_per_Kpc3
        self.to_Msun_per_Mpc3 = 1.0 / self.from_Msun_per_Mpc3
        self.to_Gev_per_cm3 = 1.0 / self.from_Gev_per_cm3
        self.to_hbar = 1.0 / self.from_hbar


def set_schroedinger_units(m22):
    """
    Factory function for Schroedinger code units (our main convention)
    """
    H0 = 100 * cosmo.h * Units.kms / Units.Mpc
    rho_m = cosmo.om * 3 * H0**2 / (8 * np.pi * Units.Grav)

    m = m22 * Units.m22

    T = (3 / 2 * cosmo.om * H0**2) ** (-1 / 2)
    L = np.sqrt(Units.hbar / m) * (3 / 2 * cosmo.om * H0**2) ** (-1 / 4)
    M = rho_m * L**3

    return Units(L, M, T)


schroedinger_units = set_schroedinger_units(1.0)
cgs_units = Units(1, 1, 1)
