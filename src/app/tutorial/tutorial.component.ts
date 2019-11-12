import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router, NavigationEnd } from '@angular/router';
import { Subscription } from 'rxjs';

import { Tutorial } from '../tutorial';
import { TOC } from '../toc';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.css']
})
export class TutorialComponent implements OnInit, OnDestroy {
  selectedTutorial: string;
  currentSelection: string;
  currentTutorialText: string;
  tutorialList: Tutorial[];
  tocContent: TOC[];
  routerSubscription: Subscription;

  constructor(private http: HttpClient,
    private router: Router,) { }

  ngOnInit() {
    this.getTutorialStructure();

    this.routerSubscription = this.router.events.subscribe((e: any) => {
      if (e instanceof NavigationEnd) {
        this.parseURL();
      }
    });
  }

  ngOnDestroy() {
    if (this.routerSubscription) {
      this.routerSubscription.unsubscribe();
    }
  }

  getTutorialStructure() {
    this.http.get('assets/tutorial/structure.json', {responseType: 'text'}).subscribe(data => {
      this.tutorialList = <Tutorial[]>JSON.parse(data);

      console.log(this.tutorialList);
      this.parseURL();
    });
  }

  private parseURL() {
    var pathComponents = this.router.url.split("/tutorials/");
    var name = "";
    if (pathComponents.length > 1) {
      name = pathComponents[1].split("#")[0];
    }
    else {
      name = "";
    }

    if (name === "") {
      this.updateCurrentTutorial(this.tutorialList[0]);
    }
    else {
      var t: Tutorial[] = this.tutorialList.filter(tutorial => tutorial.name === (name + ".md"));
      this.updateTutorialContent(t[0]);
    }
  }

  updateTutorialContent(tutorial: Tutorial) {
    window.scroll(0,0);

    this.selectedTutorial = tutorial.name;
    this.currentSelection = 'assets/tutorial/' + this.selectedTutorial;

    this.tocContent = tutorial.toc;
    this.getSelectedTutorialText();
  }

  updateCurrentTutorial(tutorial: Tutorial) {
    this.updateTutorialContent(tutorial);

    this.router.navigate(['/tutorials/' + tutorial.name.substring(0, tutorial.name.length - 3)]);
  }

  getSelectedTutorialText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentTutorialText = data;
    });
  }

  

}
