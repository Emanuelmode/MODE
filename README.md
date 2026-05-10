# 🌀 MODE Attractor Pipeline
**Framework de Legibilidad Observacional para Sistemas Dinámicos No Lineales**  
*Autor: Emanuel Duarte · Pergamino, Argentina · 2026*  
*Versión: v2.3 (Modular + SampEx + δ Semidinámico)*

---

## 📐 Arquitectura H1 / H2 / H3
| Capa | Concepto | Función |
|------|----------|---------|
| **H1** | `ε` Dinámico | Adapta la resolución local al sistema (KNN + escala media) |
| **H2** | `τ` Semidinámico | Estabiliza la reconstrucción de fase por régimen (AMI + cache) |
| **H3** | `R³` Descriptor | Cuantifica la co-estabilización observacional (gradientes RMS + pesos gaussianos) |

## 📁 Estructura del Proyecto
