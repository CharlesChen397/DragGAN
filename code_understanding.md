DragGAN 的核心代码逻辑包含在 `viz` 文件夹下的 `renderer.py` 代码文件中。
### 点追踪
点追踪通过在特征空间进行最近邻搜索，来更新控制点位置。

对每个控制点 `point`，提取以 `point` 为中心的方形特征图区域 `feat_patch`，计算该区域内每个像素的特征与初始控制点特征 `self.feat_refs[j]` 之间的 L2 距离，找到距离最小的像素点索引 `idx` ，并将其更新为新的控制点坐标 `points[j]`。
```py
# Point tracking with feature matching
with torch.no_grad():
    for j, point in enumerate(points):
        r = round(r2 / 512 * h)
        up = max(point[0] - r, 0)
        down = min(point[0] + r + 1, h)
        left = max(point[1] - r, 0)
        right = min(point[1] + r + 1, w)
        feat_patch = feat_resize[:,:,up:down,left:right]
        L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
        _, idx = torch.min(L2.view(1,-1), -1)
        width = right - left
        point = [idx.item() // width + up, idx.item() % width + left]
        points[j] = point

res.points = [[point[0], point[1]] for point in points]
```
### 运动监督
运动监督通过构建一个特殊的损失函数，使得图像内容向目标状态移动。设有 $n$ 个点对 $(\boldsymbol{p}_i,\boldsymbol{t}_i)$，损失函数为
$$
\mathcal{L} = \sum_{i=0}^{n} \sum_{\boldsymbol{q}_i \in \Omega_1 (\boldsymbol{p}_i, r_1)} \| \mathrm{F}(\boldsymbol{q}_i) - \mathrm{F}(\boldsymbol{q}_i + \boldsymbol{d}_i) \|_1 + \lambda \| (\mathrm{F} - \mathrm{F}_0) \cdot (1 - \mathbf{M}) \|_1
$$
其中 $\Omega_1 (\boldsymbol{p}_i, r_1)$ 为 $\boldsymbol{p}_i$ 的 $r_1$ 邻域，$\boldsymbol{d}_i$ 是从 $\boldsymbol{p}_i$ 指向 $\boldsymbol{t}_i$ 的单位向量，$\mathbf{M}$ 为二进制掩码。

损失函数的前半部分控制图像内容向目标状态移动，后半部分控制图像无关区域尽量保持不变。

如下的代码片段实现了运动监督。
```py
# Motion supervision
loss_motion = 0
res.stop = True
for j, point in enumerate(points):
    direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
    if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
        res.stop = False
    if torch.linalg.norm(direction) > 1:
        distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
        relis, reljs = torch.where(distance < round(r1 / 512 * h))
        direction = direction / (torch.linalg.norm(direction) + 1e-7)
        gridh = (relis+direction[1]) / (h-1) * 2 - 1
        gridw = (reljs+direction[0]) / (w-1) * 2 - 1
        grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
        target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
        loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

loss = loss_motion
if mask is not None:
    if mask.min() == 0 and mask.max() == 1:
        mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
        loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
        loss += lambda_mask * loss_fix

loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
```
### 更新
```py
if not res.stop:
    self.w_optim.zero_grad()
    loss.backward()
    self.w_optim.step()
```