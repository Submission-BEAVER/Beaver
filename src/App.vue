<script>

import Header from './components/Header.vue'
import Main from './components/Main.vue'
import Footer from './components/Footer.vue'

export default {
  components: {
    Header,
    Main,
    Footer,
  }
}

</script>

<template>
    <div class="app-root">
      
      <el-container>
        <el-header><Header/></el-header>
        <el-main><Main/></el-main>
        <el-footer><Footer/></el-footer>
      </el-container>
      
      <!-- 回到顶部 -->
      <el-backtop :right="40" :bottom="80" />
    </div>
</template>

<style>

:root {
  color: #17202a;
  background: #ffffff;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

body {
  margin: 0;
  background: #ffffff;
}

.app-root {
  min-height: 100vh;
}

.el-main {
  padding: 0;
}

.section-shell {
  margin: 0 auto;
  padding: 54px 24px 12px;
}

.section-shell + .section-shell {
  padding-top: 42px;
}

h1 {
  font-family: "MyFont", Verdana, sans-serif;
  letter-spacing: 0;
  font-size: 30px;
  line-height: 1.2;
  margin: 0 0 24px;
  text-align: center;
}

h2, h3, h4, h5, h6 {
  font-family: "MyFont", Verdana, sans-serif;
  letter-spacing: 0;
  color: #111827;
}

h1:first-child {
  position: relative;
}

h1:first-child::after {
  background: #0f5f78;
  border-radius: 999px;
  content: "";
  display: block;
  height: 3px;
  margin: 14px auto 0;
  width: 56px;
}

h3 {
  border-top: 1px solid #e7edf3;
  font-size: 24px;
  line-height: 1.25;
  margin: 54px 0 18px;
  padding-top: 30px;
}

h3:first-child {
  border-top: 0;
  margin-top: 0;
  padding-top: 0;
}

.el-header {
  padding: 0px;
}

.el-footer {
  padding: 0px;
}

p, li {
  color: #334155;
  font-size: 17px;
  line-height: 1.8;
}

p {
  margin: 0 0 18px;
}

ul {
  margin: 10px 0 24px;
  padding-left: 1.4rem;
}

li {
  margin: 8px 0;
}

code {
  font-family: 'Consolas', monospace;
  border-radius: 5px;
  padding: 2px 5px;
  background: #f2f2f2;
  color: black;
}

pre {
  border-radius: 5px;
  padding: 10px;
  background: #f2f2f2;
  line-height: 1.4rem;
}

pre:not(pre[id=bibtex]) {
  overflow-x: auto;
}

pre code {
  font-size: 16px;
  color: black;
  border: none;
  padding: 0px;
}

blockquote {
  border-left: 5px solid #bcbcbc;
  padding: 10px 20px 10px 20px;
  margin: 0px;
  background: #f6f6f6;
}

blockquote > p {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
}

/* 链接颜色装饰 */
a {
  color: #3273dc;
  text-decoration: none;
}

/* 鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: #848484;
  border-bottom: dotted;
}

p > img {
  width: 100%;
  max-width: 1060px;
  display: block;
  margin: 24px auto 30px;
  border: 1px solid #dce6ef;
  border-radius: 8px;
  box-shadow: 0 14px 34px rgba(15, 35, 55, 0.10);
}

.figure-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 26px 0 28px;
}

.figure-grid img,
.figure-stack img {
  background: #ffffff;
  border: 1px solid #dce6ef;
  border-radius: 8px;
  box-shadow: 0 12px 30px rgba(15, 35, 55, 0.10);
  display: block;
  height: auto;
  width: 100%;
}

.figure-stack {
  display: grid;
  gap: 16px;
  margin: 24px auto 28px;
}

.figure-stack.compact {
  max-width: 980px;
}

.performance-grid {
  align-items: center;
  grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.6fr);
}

table {
  border-collapse: collapse;
  width: max-content;
  max-width: 100%;
  margin: 0 auto;
  display:block;
  overflow-x:auto;
}

thead {
  border-bottom-width: 1px;
  border-top-width: 2px;
  border-left-width: 0px;
  border-right-width: 0px;
  border-style: solid;
  border-color: rgb(0 0 0);
}

tbody tr:last-child {
  border-bottom-width: 2px;
  border-top-width: 0px;
  border-left-width: 0px;
  border-right-width: 0px;
  border-style: solid;
  border-color: rgb(0 0 0);
}

th, td {
    padding-left: 1rem;
    padding-right: 1rem;
}

.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
}

@media (max-width: 860px) {
  .section-shell {
    padding-left: 18px;
    padding-right: 18px;
  }

  .figure-grid,
  .performance-grid {
    grid-template-columns: 1fr;
  }

  h3 {
    font-size: 22px;
    margin-top: 44px;
  }
}

@media (max-width: 560px) {
  .section-shell {
    padding-top: 40px;
  }

  h1 {
    font-size: 26px;
  }

  p, li {
    font-size: 16px;
  }
}

</style>
