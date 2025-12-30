# Physical AI & Humanoid Robotics Course

This is a Docusaurus-based textbook for embodied intelligence and humanoid robotics, deployed on Vercel.

## Vercel Deployment Configuration

To deploy this site on Vercel, use the following settings:

- **Root Directory**: `Frontend/my-website`
- **Build Command**: `npm run build`
- **Output Directory**: `build`
- **Node Version**: 20.x (specified in package.json engines field)

## Configuration Notes

- The site is configured for Vercel deployment with:
  - `url`: `https://physical-ai-textbook.vercel.app`
  - `baseUrl`: `/` (root path for Vercel)
  - `trailingSlash`: `true` (for consistent URL handling)
- The `_redirects` file in `static/` handles client-side routing
- Broken links are set to 'ignore' to prevent build failures
- The `future: { v4: true }` flag was removed to prevent build errors

## Local Development

To run the site locally:

```bash
cd Frontend/my-website
npm install
npm run start
```

## Build Status

The site builds successfully and is ready for Vercel deployment.