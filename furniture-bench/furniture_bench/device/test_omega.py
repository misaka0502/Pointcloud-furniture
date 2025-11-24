import forcedimension_core.containers as containers
import forcedimension_core.dhd as dhd

dhd.open()
pos = containers.Vec3()

# Equivalent to: dhd.getPosition(out=pos)
dhd.direct.getPosition(out=pos)

print(pos)