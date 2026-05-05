<script lang="ts" setup>
// 页脚需要的链接
const links = {
    Vue: "https://cn.vuejs.org/",
    Vite: "https://cn.vitejs.dev/",
    ElementPlus: "https://element-plus.org/zh-CN/",
}
</script>

<template>
  
  <div class="bg">
    
    <el-watermark :content="['BEAVER', 'NeurIPS 2026', 'Demo Page']" :gap="[60, 0]" :z-index="0">
      <el-row justify="center">
        <p class="footer">
          BEAVER is a training-free hierarchical prompt compression project page.<br/>
          
          Built with 
          <a :href="links.Vue">Vue</a>, 
          <a :href="links.Vite">Vite</a>, and
          <a :href="links.ElementPlus">Element Plus</a>.
        </p>
      </el-row>
    </el-watermark>
  </div>
</template>

<style scoped>

/* 页脚背景 */
.bg {
  background: #102235;
  color: #d8e5ee;
  padding: 20px 0;
}

/* 页脚文字 */
.footer {
  font-size: 14px;
  line-height: 1.7;
  text-align: center;
  margin: 20px;
  z-index: 1;
}

a {
  color: #ffffff;
  font-weight: 700;
}

</style>
