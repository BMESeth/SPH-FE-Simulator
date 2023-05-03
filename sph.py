# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:21:35 2023

@author: Seth Mischo
@DrPupper_RG - Twitter
@DrPupperBME - GitHub

"""

import numpy as np

class SPH:
    def __init__(self, mass, rho0, h, dt):
        self.mass = mass
        self.rho0 = rho0
        self.h = h
        self.dt = dt
        self.W = self.poly6_kernel
        self.gradW = self.spiky_kernel_gradient

    def poly6_kernel(self, r, h):
        if r > h:
            return 0
        s = (h**2 - r**2) / h**2
        return 315 / (64 * np.pi * h**9) * s**3

    def spiky_kernel_gradient(self, r, h):
        if r > h:
            return 0
        s = (h - r) / h
        return -45 / (np.pi * h**6) * s**2

    def density_pressure(self, particles):
        for p1 in particles:
            rho = 0
            for p2 in particles:
                r = np.linalg.norm(p1.pos - p2.pos)
                rho += self.mass * self.W(r, self.h)
            p1.rho = rho
            p1.p = max(0, p1.rho - self.rho0)

    def forces(self, particles):
        for p1 in particles:
            f_pressure = np.zeros(2)
            f_viscosity = np.zeros(2)
            for p2 in particles:
                if p1 == p2:
                    continue

                r = p1.pos - p2.pos
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-5:
                    continue

                gradW = self.gradW(r_norm, self.h) * r / r_norm
                f_pressure -= self.mass * (p1.p + p2.p) / (2 * p2.rho) * gradW
                f_viscosity += self.mass * (p2.vel - p1.vel) / p2.rho * self.laplacian(self.W, r_norm, self.h)

            p1.f_pressure = f_pressure
            p1.f_viscosity = f_viscosity
    def laplacian(self, W, r, h):
        if r > h:
            return 0
        s = (h - r) / h
        return 45 / (np.pi * h**6) * s
    
    def check_boundary(self, pos, vel, width, height, radius):
        if pos[0] - radius <= 0:
            pos[0] = width - radius
        elif pos[0] + radius >= width:
            pos[0] = radius
        if pos[1] - radius <= 0:
            pos[1] = height - radius
        elif pos[1] + radius >= height:
            pos[1] = radius
        return pos, vel
    
    def update(self, particles, width, height):
        self.density_pressure(particles)
        self.forces(particles)
        for p in particles:
            p.vel += self.dt * (p.f_pressure / p.rho + p.f_viscosity / p.rho + p.f_external)
            p.pos += self.dt * p.vel
#            p.pos, p.vel = self.check_boundary(p.pos, p.vel, width, height, 5)
