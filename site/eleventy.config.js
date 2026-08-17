export default function (eleventyConfig) {
  eleventyConfig.ignores.add("README.md");
  eleventyConfig.ignores.add("FORBIDDEN.txt");
  eleventyConfig.ignores.add("stamp_data.py");
  eleventyConfig.ignores.add("scripts/**");
  eleventyConfig.ignores.add("figures-web-src/**");
  eleventyConfig.ignores.add("package.json");
  eleventyConfig.ignores.add("package-lock.json");

  eleventyConfig.addPassthroughCopy("css");
  eleventyConfig.addPassthroughCopy("js");
  eleventyConfig.addPassthroughCopy("fonts");
  eleventyConfig.addPassthroughCopy("figures-web");
  eleventyConfig.addPassthroughCopy("favicon.svg");

  eleventyConfig.addShortcode("scienceFig", function (stem, alt, caption, readAs) {
    const url = eleventyConfig.getFilter("url");
    const svgPath = url(`/figures-web/${stem}.svg`);
    const pngPath = url(`/figures-web/${stem}-2x.png`);
    const read = readAs ? `<p class="read-as">${readAs}</p>` : "";
    return `<figure class="science" id="${stem}">
  <picture>
    <source type="image/svg+xml" srcset="${svgPath}">
    <img src="${pngPath}" alt="${alt}">
  </picture>
  <figcaption>${caption}</figcaption>
  ${read}
</figure>`;
  });

  return {
    pathPrefix: "/asr-gate/",
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    dir: {
      input: ".",
      includes: "_includes",
      data: "_data",
      output: "_site",
    },
  };
}
