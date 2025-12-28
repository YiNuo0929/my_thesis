import matplotlib.pyplot as plt

# 資料
z_dim = [8, 16, 32, 64, 128, 256, 512]
mde = [2.0603, 1.6557, 1.6472, 1.5680, 1.7800, 2.0163, 2.3544]

# 等距 x 軸（用 index）
x = range(len(z_dim))

plt.figure()
plt.plot(x, mde, marker='o')

# 用 z_dim 當刻度標籤
plt.xticks(x, z_dim)

plt.xlabel('z_dim (Dimension)')
plt.ylabel('MDE (meters)')
plt.title('Bottleneck Test: z_dim vs MDE')
plt.grid(True)

plt.savefig('bottleneck_mde_equal_spacing.png', dpi=300, bbox_inches='tight')
plt.show()