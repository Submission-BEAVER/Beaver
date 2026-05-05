<script lang="ts" setup>

import { SuccessFilled } from '@element-plus/icons-vue'

// 项目主页
const logo = {
    name: "BEAVER",
    link: "#",
}

</script>

<template>
    <!-- 设置页首始终悬浮顶部 -->
    <el-affix :offset="0" :style="{width: '100%'}">
    
    <!-- 水平导航目录 -->
    <el-menu 
        default-active="1"
        class="el-menu-demo"
        mode="horizontal"
        :ellipsis="false"
    >
        <!-- 项目主页 -->
        <el-menu-item index="1">
            <el-icon :size="20"><SuccessFilled /></el-icon>
            <a :href=logo.link>{{ logo.name }} </a>
        </el-menu-item>

    </el-menu>
    </el-affix>
</template>

<style>

/* 导航背景 */
.el-menu {
	background: rgba(255, 255, 255, 0.92);
}

/* 下拉菜单宽度 */
.el-menu--popup {
	min-width: 100%;
}

/* 导航背景特效和阴影 */
.el-affix--fixed {
	box-shadow: 0 1px 12px rgba(15, 35, 55, 0.08);
	background: rgba(255, 255, 255, 0.92);
	backdrop-filter: blur(10px);
}

/* 水平导航高度 */
.el-header {
    --el-header-height: 50px;
}

</style>

<style scoped>

/* 水平导航高度 */
.el-menu--horizontal {
    --el-menu-horizontal-height: 50px;
    border-bottom: 1px solid #edf2f7;
    justify-content: center;
}

.el-menu-item {
  color: #172535;
  font-weight: 800;
}

.el-menu-item.is-active {
  border-bottom-color: #0f5f78;
}

/* 取消鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: inherit;
  border-bottom: none;
}

/* 链接装饰，取消下划线和链接颜色 */
a {
	text-decoration: None;
	color: inherit;
}

</style>

  
