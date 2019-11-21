# FastestimatorWeb

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 8.1.1.

## prerequisite

1. Install [node.js](https://nodejs.org/en/)
2. Run `npm install -g @angular/cli` to install [Angular CLI]

## Setup

Run `npm install` in the repo folder to install all angular dependency.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

## Code scaffolding

Run `ng generate component component-name` to generate a new component. You can also use `ng generate directive|pipe|service|class|guard|interface|enum|module`.

## Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory. Use the `--prod` flag for a production build.

## Deploy

### github page

Build the website with prod settings using the following command: `ng build --prod --base-href "https://fastestimator.org/"`.   
Once build is done, use `ngh` to automatically push to gh-pages branch. Then create a pull request to deploy to prod.

### firebase

1. create a firebase account
2. install firebase cli by running `sudo npm i -g firebase-tools`
3. login by running `firebase login`
4. build the website by running `ng build --prod` (no need for --base-href)
5. init `firebase init`
6. test the web before deployment `firebase serve`
7. deploy web app `firebase deploy`


## Running unit tests

Run `ng test` to execute the unit tests via [Karma](https://karma-runner.github.io).

## Running end-to-end tests

Run `ng e2e` to execute the end-to-end tests via [Protractor](http://www.protractortest.org/).

## Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI README](https://github.com/angular/angular-cli/blob/master/README.md).


